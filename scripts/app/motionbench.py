import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import av
import cv2
import joblib
import mediapipe as mp
import numpy as np
import streamlit as st
import torch
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate.rep_counting_methods import EXERCISE_CONFIGS, FixedThresholdFSMCounter, SmoothingBuffer, extract_primary_angle, normalize_exercise_name
from scripts.realtime_eval.evaluate_realtime_webcam import (
    MODEL_SPECS,
    build_landmark_indices,
    build_model_and_tools,
    extract_frame_features,
    get_angle_triplets,
    load_pose_module,
)


SEQUENCE_LENGTH = 30
FEATURE_COUNT = 78
DEFAULT_MODELS_ROOT = "models"
DEFAULT_PREDICTION_INTERVAL = 1.0
CAMERA_INDEX_CANDIDATES = [0, 1, 2]
RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
BROWSER_MEDIA_CONSTRAINTS = {"video": {"width": {"ideal": 1280}, "height": {"ideal": 720}, "frameRate": {"ideal": 24, "max": 30}}, "audio": False}


def load_runtime(model_name: str, models_root: str, feature_count: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = SimpleNamespace(model_name=model_name, models_root=models_root, feature_count=feature_count)
    model, scaler, label_encoder = build_model_and_tools(args, device)
    return device, model, scaler, label_encoder


def load_similarity_asset(model_name: str, models_root: str):
    asset_path = Path(models_root) / model_name / "weights" / "similarity_centroids.pkl"
    if not asset_path.exists():
        return None
    return joblib.load(asset_path)


def cosine_similarity_percent(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    denom = float(np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
    if denom <= 1e-8:
        return 0.0
    score = float(np.dot(vector_a, vector_b) / denom)
    score = max(-1.0, min(1.0, score))
    return ((score + 1.0) / 2.0) * 100.0


def read_valid_frame(capture: cv2.VideoCapture, max_reads: int = 20) -> np.ndarray | None:
    frame_bgr = None
    for _ in range(max_reads):
        ok, candidate = capture.read()
        if not ok:
            continue
        if float(np.mean(candidate)) > 5.0:
            return candidate
        frame_bgr = candidate
    if frame_bgr is not None and float(np.mean(frame_bgr)) > 5.0:
        return frame_bgr
    return None


def open_camera_with_fallback() -> cv2.VideoCapture | None:
    for camera_index in CAMERA_INDEX_CANDIDATES:
        capture = cv2.VideoCapture(camera_index)
        if not capture.isOpened():
            capture.release()
            continue
        frame_bgr = read_valid_frame(capture, max_reads=10)
        if frame_bgr is not None:
            return capture
        capture.release()
    return None


def create_runtime_state():
    return {
        "counter": None,
        "smoother": None,
        "active_exercise": None,
        "current_label": "none",
        "current_similarity": None,
        "current_reps": 0,
        "last_prediction_time": 0.0,
        "window": [],
    }


def process_single_frame(frame_bgr: np.ndarray, state: dict, model, scaler, label_encoder, device, pose_estimator, pose_module, landmark_indices, angle_triplets, similarity_asset, prediction_interval: float):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pose = pose_estimator.process(frame_rgb)

    drawing_utils = mp.solutions.drawing_utils
    if drawing_utils is not None and pose.pose_landmarks:
        drawing_utils.draw_landmarks(frame_bgr, pose.pose_landmarks, pose_module.POSE_CONNECTIONS)

    frame_features = extract_frame_features(pose, landmark_indices, angle_triplets)
    if frame_features is not None:
        state["window"].append(frame_features)
        if len(state["window"]) > SEQUENCE_LENGTH:
            state["window"].pop(0)

    if len(state["window"]) == SEQUENCE_LENGTH and (time.time() - state["last_prediction_time"]) >= prediction_interval:
        now = time.time()
        sequence_flat = np.array(state["window"], dtype=np.float32).reshape(1, -1)
        scaled_flat = scaler.transform(sequence_flat)
        scaled = scaled_flat.reshape(1, SEQUENCE_LENGTH, FEATURE_COUNT)
        input_tensor = torch.tensor(scaled, dtype=torch.float32, device=device)
        with torch.inference_mode():
            logits = model(input_tensor)
            prediction_index = int(torch.argmax(logits, dim=1).item())
            predicted_label = label_encoder.classes_[prediction_index]
        state["current_label"] = predicted_label

        if similarity_asset is not None:
            scaled_vector = scaled_flat[0]
            centroids = similarity_asset.get("centroids", {})
            centroid_vector = centroids.get(state["current_label"])
            if centroid_vector is not None:
                state["current_similarity"] = cosine_similarity_percent(scaled_vector.astype(np.float32), np.asarray(centroid_vector, dtype=np.float32))
            else:
                state["current_similarity"] = None
        state["last_prediction_time"] = now

    normalized_label = normalize_exercise_name(state["current_label"])
    current_reps = 0
    if pose.pose_landmarks and normalized_label in EXERCISE_CONFIGS:
        if normalized_label != state["active_exercise"]:
            config = EXERCISE_CONFIGS[normalized_label]
            state["counter"] = FixedThresholdFSMCounter(config.fixed_low, config.fixed_high, config.min_state_frames)
            state["smoother"] = SmoothingBuffer(config.smoothing_window)
            state["active_exercise"] = normalized_label

        landmarks = {}
        for name, index in landmark_indices.items():
            lm = pose.pose_landmarks.landmark[index]
            landmarks[name] = np.array([lm.x, lm.y, lm.z], dtype=np.float32) if lm.visibility >= 0.5 else np.array([0.0, 0.0, 0.0], dtype=np.float32)

        config = EXERCISE_CONFIGS[normalized_label]
        raw_angle = extract_primary_angle(landmarks, config)
        smoothed_angle = state["smoother"].update(raw_angle)
        state["counter"].update(smoothed_angle)
        current_reps = state["counter"].reps
    else:
        state["active_exercise"] = None
        state["counter"] = None
        state["smoother"] = None

    state["current_reps"] = current_reps
    show_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return show_frame, state


class MotionVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.initialized = False
        self.state = create_runtime_state()

    def configure(self, _model_name, model, scaler, label_encoder, device, pose_module, landmark_indices, angle_triplets, similarity_asset, prediction_interval):
        self.model = model
        self.scaler = scaler
        self.label_encoder = label_encoder
        self.device = device
        self.pose_module = pose_module
        self.landmark_indices = landmark_indices
        self.angle_triplets = angle_triplets
        self.similarity_asset = similarity_asset
        self.prediction_interval = prediction_interval
        self.pose_estimator = pose_module.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.initialized = True

    def recv(self, frame):
        frame_bgr = frame.to_ndarray(format="bgr24")
        with self.lock:
            if self.initialized:
                frame_rgb, self.state = process_single_frame(
                    frame_bgr=frame_bgr,
                    state=self.state,
                    model=self.model,
                    scaler=self.scaler,
                    label_encoder=self.label_encoder,
                    device=self.device,
                    pose_estimator=self.pose_estimator,
                    pose_module=self.pose_module,
                    landmark_indices=self.landmark_indices,
                    angle_triplets=self.angle_triplets,
                    similarity_asset=self.similarity_asset,
                    prediction_interval=self.prediction_interval,
                )
            else:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")

    def __del__(self):
        if hasattr(self, "pose_estimator"):
            self.pose_estimator.close()


def render_metrics(model_name: str, state: dict, slot):
    similarity_text = f"{state['current_similarity']:0.1f}%" if state["current_similarity"] is not None else "N/A"
    slot.markdown(f"### Live Metrics\nModel: `{model_name}`\n\nExercise: `{state['current_label']}`\n\nReps: `{state['current_reps']}`\n\nSimilarity: `{similarity_text}`")


def inject_browser_webrtc_css():
    st.markdown(
        """
<style>
div[data-testid="stHorizontalBlock"] {
    align-items: stretch;
}
div[data-testid="stHorizontalBlock"] > div:first-child {
    min-height: 560px;
}
div[data-testid="stHorizontalBlock"] > div:first-child iframe {
    width: 100% !important;
    min-height: 560px !important;
}
div[data-testid="stHorizontalBlock"] > div:first-child video {
    width: 100% !important;
    height: auto !important;
    min-height: 560px;
    max-height: 560px;
    object-fit: cover;
    background: #000;
    border-radius: 0.5rem;
    pointer-events: none;
}
div[data-testid="stHorizontalBlock"] > div:first-child video::-webkit-media-controls {
    display: none !important;
}
div[data-testid="stHorizontalBlock"] > div:first-child video::-webkit-media-controls-panel {
    display: none !important;
}
div[data-testid="stHorizontalBlock"] > div:first-child video::-webkit-media-controls-play-button {
    display: none !important;
}
div[data-testid="stHorizontalBlock"] > div:first-child video::-webkit-media-controls-timeline {
    display: none !important;
}
div[data-testid="stHorizontalBlock"] > div:first-child video::-webkit-media-controls-current-time-display {
    display: none !important;
}
div[data-testid="stHorizontalBlock"] > div:first-child video::-webkit-media-controls-time-remaining-display {
    display: none !important;
}
div[data-testid="stHorizontalBlock"] > div:first-child video::-webkit-media-controls-mute-button {
    display: none !important;
}
div[data-testid="stHorizontalBlock"] > div:first-child video::-webkit-media-controls-toggle-closed-captions-button {
    display: none !important;
}
div[data-testid="stHorizontalBlock"] > div:first-child video::-webkit-media-controls-fullscreen-button {
    display: none !important;
}
div[data-testid="stHorizontalBlock"] > div:first-child button[kind="secondary"] {
    display: none !important;
}
</style>
""",
        unsafe_allow_html=True,
    )


def run_local_session(model_name: str, model, scaler, label_encoder, device, pose_module, landmark_indices, angle_triplets, similarity_asset):
    capture = open_camera_with_fallback()
    if capture is None:
        st.error("No camera device found in runtime. If using Docker locally, run with --device=/dev/video0:/dev/video0. On Hugging Face Spaces, use Browser (HF/Cloud) mode.")
        st.session_state.session_active = False
        return

    left_col, right_col = st.columns([2, 1])
    with left_col:
        frame_slot = st.empty()
    with right_col:
        metrics_slot = st.empty()

    state = create_runtime_state()
    with pose_module.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_estimator:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            show_frame, state = process_single_frame(
                frame_bgr=frame_bgr,
                state=state,
                model=model,
                scaler=scaler,
                label_encoder=label_encoder,
                device=device,
                pose_estimator=pose_estimator,
                pose_module=pose_module,
                landmark_indices=landmark_indices,
                angle_triplets=angle_triplets,
                similarity_asset=similarity_asset,
                prediction_interval=DEFAULT_PREDICTION_INTERVAL,
            )
            frame_slot.image(show_frame, channels="RGB", width="stretch")
            render_metrics(model_name, state, metrics_slot)
            if not st.session_state.session_active:
                break

    capture.release()
    if st.session_state.session_active:
        st.session_state.session_active = False
        st.session_state.session_notice = "finished"
        st.rerun()


def run_browser_session(model_name: str, model, scaler, label_encoder, device, pose_module, landmark_indices, angle_triplets, similarity_asset):
    inject_browser_webrtc_css()
    left_col, right_col = st.columns([2, 1])
    with left_col:
        webrtc_ctx = webrtc_streamer(
            key="motionbench-webrtc",
            video_processor_factory=MotionVideoProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints=BROWSER_MEDIA_CONSTRAINTS,
            desired_playing_state=st.session_state.session_active,
            video_html_attrs={"controls": False, "autoPlay": True, "muted": True, "playsInline": True},
            async_processing=True,
        )
    with right_col:
        metrics_slot = st.empty()

    if webrtc_ctx.video_processor:
        video_processor = webrtc_ctx.video_processor
        if not video_processor.initialized:
            video_processor.configure(model_name, model, scaler, label_encoder, device, pose_module, landmark_indices, angle_triplets, similarity_asset, DEFAULT_PREDICTION_INTERVAL)
        with video_processor.lock:
            state_copy = {
                "current_label": video_processor.state["current_label"],
                "current_reps": video_processor.state["current_reps"],
                "current_similarity": video_processor.state["current_similarity"],
            }
        render_metrics(model_name, state_copy, metrics_slot)
    else:
        metrics_slot.info("Allow camera access, then choose your device under Video Input below the feed if needed.")


def main():
    st.set_page_config(page_title="MotionBench", layout="wide")
    st.title("MotionBench Live")

    if "session_active" not in st.session_state:
        st.session_state.session_active = False
    if "session_notice" not in st.session_state:
        st.session_state.session_notice = None

    is_hf_space = bool(os.getenv("SPACE_ID"))
    model_name = st.selectbox("Select Model", options=list(MODEL_SPECS.keys()), index=0)
    camera_options = ["Browser (HF/Cloud)", "Local OpenCV (Desktop)"]
    default_camera_index = 0 if is_hf_space else 1
    camera_source = st.selectbox("Camera Source", options=camera_options, index=default_camera_index)
    session_button_label = "Stop Session" if st.session_state.session_active else "Start Session"
    session_button_clicked = st.button(session_button_label, width="stretch")

    if session_button_clicked and st.session_state.session_active:
        st.session_state.session_active = False
        st.session_state.session_notice = "stopped"
        st.rerun()

    if session_button_clicked and not st.session_state.session_active:
        st.session_state.session_active = True
        st.session_state.session_notice = None
        st.rerun()

    if not st.session_state.session_active:
        if st.session_state.session_notice == "stopped":
            st.info("Session stopped.")
            st.session_state.session_notice = None
        elif st.session_state.session_notice == "finished":
            st.success("Session finished.")
            st.session_state.session_notice = None
        st.info("Select a model, then start session.")
        return

    device, model, scaler, label_encoder = load_runtime(model_name, DEFAULT_MODELS_ROOT, feature_count=FEATURE_COUNT)
    similarity_asset = load_similarity_asset(model_name, DEFAULT_MODELS_ROOT)
    pose_module = load_pose_module()
    landmark_indices = build_landmark_indices(pose_module)
    angle_triplets = get_angle_triplets()

    if camera_source == "Local OpenCV (Desktop)":
        run_local_session(model_name, model, scaler, label_encoder, device, pose_module, landmark_indices, angle_triplets, similarity_asset)
    else:
        run_browser_session(model_name, model, scaler, label_encoder, device, pose_module, landmark_indices, angle_triplets, similarity_asset)


if __name__ == "__main__":
    main()
