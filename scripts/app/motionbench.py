import time
from collections import Counter, deque
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import joblib
import numpy as np
import streamlit as st
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate.rep_counting_methods import (
    EXERCISE_CONFIGS,
    FixedThresholdFSMCounter,
    SmoothingBuffer,
    extract_primary_angle,
    normalize_exercise_name,
)
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
LABEL_SMOOTHING_WINDOW = 5
DEFAULT_MODELS_ROOT = "models"
DEFAULT_CAMERA_INDEX = 0
DEFAULT_PREDICTION_INTERVAL = 1.0
CAMERA_INDEX_CANDIDATES = [0, 1, 2]


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


def smooth_label(label_history: deque[str]) -> str:
    if not label_history:
        return "none"
    counts = Counter(label_history)
    return counts.most_common(1)[0][0]


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


def test_camera() -> tuple[bool, np.ndarray | None]:
    for camera_index in CAMERA_INDEX_CANDIDATES:
        capture = cv2.VideoCapture(camera_index)
        if not capture.isOpened():
            capture.release()
            continue
        frame_bgr = read_valid_frame(capture)
        capture.release()
        if frame_bgr is not None:
            return True, cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return False, None


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


def main():
    st.set_page_config(page_title="MotionBench", layout="wide")
    st.title("MotionBench Live")

    if "session_active" not in st.session_state:
        st.session_state.session_active = False

    model_name = st.selectbox("Select Model", options=list(MODEL_SPECS.keys()), index=0)
    controls_col_a, controls_col_b = st.columns(2)
    with controls_col_a:
        camera_test_clicked = st.button("Test Camera", width="stretch")
    with controls_col_b:
        start_clicked = st.button("Start Session", width="stretch")

    if camera_test_clicked:
        ok, preview_frame = test_camera()
        if ok:
            st.image(preview_frame, channels="RGB", width="stretch")
        else:
            st.error("Could not capture a valid camera frame.")

    if start_clicked:
        st.session_state.session_active = True

    if not st.session_state.session_active:
        st.info("Select a model, test camera, then start session.")
        return

    device, model, scaler, label_encoder = load_runtime(model_name, DEFAULT_MODELS_ROOT, feature_count=FEATURE_COUNT)
    similarity_asset = load_similarity_asset(model_name, DEFAULT_MODELS_ROOT)
    pose_module = load_pose_module()
    landmark_indices = build_landmark_indices(pose_module)
    angle_triplets = get_angle_triplets()

    capture = open_camera_with_fallback()
    if capture is None:
        st.error("Could not capture a valid camera frame.")
        st.session_state.session_active = False
        return

    left_col, right_col = st.columns([2, 1])
    with right_col:
        stop_clicked = st.button("Stop Session", width="stretch")
    if stop_clicked:
        st.session_state.session_active = False
        capture.release()
        st.info("Session stopped.")
        return

    with left_col:
        frame_slot = st.empty()
    with right_col:
        metrics_slot = st.empty()

    rep_counter = None
    rep_smoother = None
    active_exercise_label = None
    current_label = "none"
    current_similarity = None
    last_prediction_time = 0.0
    label_history = deque(maxlen=LABEL_SMOOTHING_WINDOW)
    window = []
    prediction_interval = DEFAULT_PREDICTION_INTERVAL

    with pose_module.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose_estimator:
        drawing_utils = None
        try:
            import mediapipe as mp

            drawing_utils = mp.solutions.drawing_utils
        except Exception:
            drawing_utils = None

        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            elapsed = time.time()

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pose_result = pose_estimator.process(frame_rgb)

            if drawing_utils is not None and pose_result.pose_landmarks:
                drawing_utils.draw_landmarks(
                    frame_bgr,
                    pose_result.pose_landmarks,
                    pose_module.POSE_CONNECTIONS,
                )

            frame_features = extract_frame_features(pose_result, landmark_indices, angle_triplets)

            if frame_features is not None:
                window.append(frame_features)
                if len(window) > SEQUENCE_LENGTH:
                    window.pop(0)

            if len(window) == SEQUENCE_LENGTH and (time.time() - last_prediction_time) >= prediction_interval:
                sequence_array = np.array(window, dtype=np.float32).reshape(1, -1)
                scaled_flat = scaler.transform(sequence_array)
                scaled = scaled_flat.reshape(1, SEQUENCE_LENGTH, FEATURE_COUNT)
                input_tensor = torch.tensor(scaled, dtype=torch.float32, device=device)
                with torch.inference_mode():
                    logits = model(input_tensor)
                    prediction_index = int(torch.argmax(logits, dim=1).item())
                    predicted_label = label_encoder.classes_[prediction_index]
                label_history.append(predicted_label)
                current_label = smooth_label(label_history)
                if similarity_asset is not None:
                    scaled_vector = scaled_flat[0]
                    centroids = similarity_asset.get("centroids", {})
                    centroid_vector = centroids.get(current_label)
                    if centroid_vector is not None:
                        current_similarity = cosine_similarity_percent(
                            scaled_vector.astype(np.float32),
                            np.asarray(centroid_vector, dtype=np.float32),
                        )
                    else:
                        current_similarity = None
                last_prediction_time = time.time()

            normalized_label = normalize_exercise_name(current_label)
            current_reps = 0
            if pose_result.pose_landmarks and normalized_label in EXERCISE_CONFIGS:
                if normalized_label != active_exercise_label:
                    config = EXERCISE_CONFIGS[normalized_label]
                    rep_counter = FixedThresholdFSMCounter(config.fixed_low, config.fixed_high, config.min_state_frames)
                    rep_smoother = SmoothingBuffer(config.smoothing_window)
                    active_exercise_label = normalized_label

                landmarks = {}
                for name, index in landmark_indices.items():
                    lm = pose_result.pose_landmarks.landmark[index]
                    landmarks[name] = np.array([lm.x, lm.y, lm.z], dtype=np.float32) if lm.visibility >= 0.5 else np.array([0.0, 0.0, 0.0], dtype=np.float32)

                config = EXERCISE_CONFIGS[normalized_label]
                raw_angle = extract_primary_angle(landmarks, config)
                smoothed_angle = rep_smoother.update(raw_angle)
                rep_counter.update(smoothed_angle)
                current_reps = rep_counter.reps
            else:
                active_exercise_label = None
                rep_counter = None
                rep_smoother = None

            show_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_slot.image(show_frame, channels="RGB", width="stretch")
            similarity_text = f"{current_similarity:0.1f}%" if current_similarity is not None else "N/A"
            metrics_slot.markdown(
                f"### Live Metrics\n"
                f"Model: `{model_name}`\n\n"
                f"Exercise: `{current_label}`\n\n"
                f"Reps: `{current_reps}`\n\n"
                f"Similarity: `{similarity_text}`"
            )

            if not st.session_state.session_active:
                break

    capture.release()
    st.session_state.session_active = False
    st.success("Session finished.")


if __name__ == "__main__":
    main()
