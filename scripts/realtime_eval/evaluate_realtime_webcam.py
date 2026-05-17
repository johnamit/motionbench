import argparse
import time
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
import torch
from torch import nn

from scripts.evaluate.rep_counting_methods import (
    EXERCISE_CONFIGS,
    FixedThresholdFSMCounter,
    SmoothingBuffer,
    extract_primary_angle,
    normalize_exercise_name,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", choices=["bilstm", "lstm", "gru", "tcn", "cnn_bilstm", "st_gcn"], required=True)
    parser.add_argument("--models-root", default="models")
    parser.add_argument("--output-dir", default="results/eval_realtime")
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--feature-count", type=int, default=78)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--run-seconds", type=int, default=75)
    parser.add_argument("--prediction-interval", type=float, default=1.0)
    return parser.parse_args()


class BidirectionalLstmClassifier(nn.Module):
    def __init__(self, feature_count, hidden_size, class_count, dropout_probability):
        super().__init__()
        self.bilstm = nn.LSTM(input_size=feature_count, hidden_size=hidden_size, num_layers=2, batch_first=True, dropout=dropout_probability, bidirectional=True)
        self.dropout = nn.Dropout(dropout_probability)
        self.classifier = nn.Linear(hidden_size * 2, class_count)

    def forward(self, input_sequence):
        recurrent_output, _ = self.bilstm(input_sequence)
        final_timestep_output = recurrent_output[:, -1, :]
        return self.classifier(self.dropout(final_timestep_output))


class LstmClassifier(nn.Module):
    def __init__(self, feature_count, hidden_size, class_count, dropout_probability):
        super().__init__()
        self.lstm = nn.LSTM(input_size=feature_count, hidden_size=hidden_size, num_layers=2, batch_first=True, dropout=dropout_probability, bidirectional=False)
        self.dropout = nn.Dropout(dropout_probability)
        self.classifier = nn.Linear(hidden_size, class_count)

    def forward(self, input_sequence):
        recurrent_output, _ = self.lstm(input_sequence)
        final_timestep_output = recurrent_output[:, -1, :]
        return self.classifier(self.dropout(final_timestep_output))


class GruClassifier(nn.Module):
    def __init__(self, feature_count, hidden_size, class_count, dropout_probability):
        super().__init__()
        self.gru = nn.GRU(input_size=feature_count, hidden_size=hidden_size, num_layers=2, batch_first=True, dropout=dropout_probability, bidirectional=False)
        self.dropout = nn.Dropout(dropout_probability)
        self.classifier = nn.Linear(hidden_size, class_count)

    def forward(self, input_sequence):
        recurrent_output, _ = self.gru(input_sequence)
        final_timestep_output = recurrent_output[:, -1, :]
        return self.classifier(self.dropout(final_timestep_output))


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, input_tensor):
        if self.chomp_size == 0:
            return input_tensor
        return input_tensor[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(input_channels, output_channels, kernel_size, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(output_channels, output_channels, kernel_size, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(input_channels, output_channels, kernel_size=1) if input_channels != output_channels else None
        self.final_relu = nn.ReLU()

    def forward(self, input_tensor):
        output_tensor = self.dropout1(self.relu1(self.chomp1(self.conv1(input_tensor))))
        output_tensor = self.dropout2(self.relu2(self.chomp2(self.conv2(output_tensor))))
        residual_tensor = input_tensor if self.downsample is None else self.downsample(input_tensor)
        return self.final_relu(output_tensor + residual_tensor)


class TcnClassifier(nn.Module):
    def __init__(self, feature_count, class_count, channel_width, kernel_size, dropout):
        super().__init__()
        self.input_projection = nn.Conv1d(feature_count, channel_width, kernel_size=1)
        self.block1 = TemporalBlock(channel_width, channel_width, kernel_size, dilation=1, dropout=dropout)
        self.block2 = TemporalBlock(channel_width, channel_width, kernel_size, dilation=2, dropout=dropout)
        self.block3 = TemporalBlock(channel_width, channel_width, kernel_size, dilation=4, dropout=dropout)
        self.classifier = nn.Linear(channel_width, class_count)

    def forward(self, input_sequence):
        temporal_tensor = input_sequence.transpose(1, 2)
        temporal_tensor = self.block3(self.block2(self.block1(self.input_projection(temporal_tensor))))
        return self.classifier(temporal_tensor[:, :, -1])


class CnnBiLstmClassifier(nn.Module):
    def __init__(self, feature_count, class_count, cnn_filters, cnn_kernel_size, lstm_units, dropout_probability):
        super().__init__()
        self.conv1d = nn.Conv1d(feature_count, cnn_filters, kernel_size=cnn_kernel_size, padding=cnn_kernel_size // 2)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_probability)
        self.bilstm = nn.LSTM(input_size=cnn_filters, hidden_size=lstm_units, num_layers=2, batch_first=True, dropout=dropout_probability, bidirectional=True)
        self.dropout2 = nn.Dropout(dropout_probability)
        self.classifier = nn.Linear(lstm_units * 2, class_count)

    def forward(self, input_sequence):
        temporal_tensor = self.dropout1(self.relu(self.conv1d(input_sequence.transpose(1, 2)))).transpose(1, 2)
        recurrent_output, _ = self.bilstm(temporal_tensor)
        return self.classifier(self.dropout2(recurrent_output[:, -1, :]))


class GraphConvolution(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.projection = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, input_tensor, adjacency_matrix):
        return torch.einsum("nctv,vw->nctw", self.projection(input_tensor), adjacency_matrix)


class StGcnBlock(nn.Module):
    def __init__(self, input_channels, output_channels, dropout):
        super().__init__()
        self.graph_convolution = GraphConvolution(input_channels, output_channels)
        self.temporal_convolution = nn.Sequential(
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=(9, 1), padding=(4, 0)),
            nn.BatchNorm2d(output_channels),
            nn.Dropout(dropout),
        )
        self.residual = nn.Sequential(nn.Conv2d(input_channels, output_channels, kernel_size=1), nn.BatchNorm2d(output_channels)) if input_channels != output_channels else nn.Identity()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, input_tensor, adjacency_matrix):
        residual_tensor = self.residual(input_tensor)
        output_tensor = self.temporal_convolution(self.graph_convolution(input_tensor, adjacency_matrix))
        return self.activation(output_tensor + residual_tensor)


class StGcnClassifier(nn.Module):
    def __init__(self, feature_count, class_count, dropout):
        super().__init__()
        self.input_batch_norm = nn.BatchNorm1d(feature_count)
        self.register_parameter("adjacency_logits", nn.Parameter(torch.eye(feature_count)))
        self.block1 = StGcnBlock(1, 64, dropout)
        self.block2 = StGcnBlock(64, 64, dropout)
        self.block3 = StGcnBlock(64, 128, dropout)
        self.classifier = nn.Linear(128, class_count)

    def forward(self, input_sequence):
        batch_size, sequence_length, feature_count = input_sequence.shape
        normalized = self.input_batch_norm(input_sequence.reshape(batch_size * sequence_length, feature_count)).reshape(batch_size, sequence_length, feature_count)
        graph_tensor = normalized.unsqueeze(1)
        adjacency = torch.softmax(self.adjacency_logits, dim=1)
        graph_tensor = self.block3(self.block2(self.block1(graph_tensor, adjacency), adjacency), adjacency)
        pooled = graph_tensor.mean(dim=2).mean(dim=2)
        return self.classifier(pooled)


MODEL_SPECS = {
    "bilstm": {"weight": "bidirectionallstm_model.pt", "scaler": "bidirectionallstm_scaler.pkl", "encoder": "bidirectionallstm_label_encoder.pkl", "builder": lambda f, c: BidirectionalLstmClassifier(f, 73, c, 0.2174)},
    "lstm": {"weight": "lstm_model.pt", "scaler": "lstm_scaler.pkl", "encoder": "lstm_label_encoder.pkl", "builder": lambda f, c: LstmClassifier(f, 117, c, 0.3829)},
    "gru": {"weight": "gru_model.pt", "scaler": "gru_scaler.pkl", "encoder": "gru_label_encoder.pkl", "builder": lambda f, c: GruClassifier(f, 96, c, 0.2)},
    "tcn": {"weight": "tcn_model.pt", "scaler": "tcn_scaler.pkl", "encoder": "tcn_label_encoder.pkl", "builder": lambda f, c: TcnClassifier(f, c, 128, 3, 0.2)},
    "cnn_bilstm": {"weight": "cnn_bilstm_model.pt", "scaler": "cnn_bilstm_scaler.pkl", "encoder": "cnn_bilstm_label_encoder.pkl", "builder": lambda f, c: CnnBiLstmClassifier(f, c, 128, 3, 73, 0.2)},
    "st_gcn": {"weight": "st_gcn_model.pt", "scaler": "st_gcn_scaler.pkl", "encoder": "st_gcn_label_encoder.pkl", "builder": lambda f, c: StGcnClassifier(f, c, 0.2)},
}


def load_pose_module():
    try:
        return mp.solutions.pose
    except AttributeError:
        from mediapipe.python.solutions import pose as pose_module
        return pose_module


def build_landmark_indices(mp_pose):
    names = [
        "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
        "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST", "LEFT_ANKLE", "RIGHT_ANKLE",
        "LEFT_HEEL", "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX", "LEFT_PINKY", "RIGHT_PINKY",
        "LEFT_INDEX", "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB"
    ]
    return {name: mp_pose.PoseLandmark[name].value for name in names}


def get_angle_triplets():
    return [
        ("LEFT_HIP", "LEFT_SHOULDER", "LEFT_ELBOW"), ("RIGHT_HIP", "RIGHT_SHOULDER", "RIGHT_ELBOW"),
        ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"), ("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),
        ("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"), ("RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"),
        ("LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE"), ("RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE"),
        ("LEFT_KNEE", "LEFT_ANKLE", "LEFT_HEEL"), ("RIGHT_KNEE", "RIGHT_ANKLE", "RIGHT_HEEL"),
        ("LEFT_ANKLE", "LEFT_HEEL", "LEFT_FOOT_INDEX"), ("RIGHT_ANKLE", "RIGHT_HEEL", "RIGHT_FOOT_INDEX"),
    ]


def calculate_angle_degrees(point_a, point_b, point_c):
    if np.allclose(point_a, 0.0) or np.allclose(point_b, 0.0) or np.allclose(point_c, 0.0):
        return 0.0
    vector_ab = point_a[:2] - point_b[:2]
    vector_cb = point_c[:2] - point_b[:2]
    denominator = np.linalg.norm(vector_ab) * np.linalg.norm(vector_cb)
    if denominator == 0.0:
        return 0.0
    cosine_value = np.clip(np.dot(vector_ab, vector_cb) / denominator, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine_value)))


def extract_frame_features(results, landmark_indices, angle_triplets, min_visibility=0.5):
    if not results.pose_landmarks:
        return None
    points = {}
    for name, idx in landmark_indices.items():
        lm = results.pose_landmarks.landmark[idx]
        if lm.visibility >= min_visibility:
            points[name] = np.array([lm.x, lm.y, lm.z], dtype=np.float32)
        else:
            points[name] = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    features = []
    for name in landmark_indices:
        point = points[name]
        features.extend([point[0], point[1], point[2]])
    for a, b, c in angle_triplets:
        features.append(calculate_angle_degrees(points[a], points[b], points[c]))
    return np.array(features, dtype=np.float32)


def build_model_and_tools(args, device):
    model_name = args.model_name
    spec = MODEL_SPECS[model_name]
    weights_dir = Path(args.models_root) / model_name / "weights"

    scaler = joblib.load(weights_dir / spec["scaler"])
    label_encoder = joblib.load(weights_dir / spec["encoder"])
    class_count = len(label_encoder.classes_)

    model = spec["builder"](args.feature_count, class_count).to(device)
    model.load_state_dict(torch.load(weights_dir / spec["weight"], map_location=device))
    model.eval()
    return model, scaler, label_encoder


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, scaler, label_encoder = build_model_and_tools(args, device)

    pose_module = load_pose_module()
    landmark_indices = build_landmark_indices(pose_module)
    angle_triplets = get_angle_triplets()

    capture = cv2.VideoCapture(args.camera_index)
    if not capture.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("Realtime evaluation started.")
    print("Protocol: 0-20s exercise A, 20-40s exercise B, 40-60s exercise C, 60-75s free.")

    window = []
    events = []
    prediction_latencies_ms = []
    frame_times = []
    predicted_labels = []

    last_prediction_time = 0.0
    current_label = "none"
    rep_counters = {}
    rep_smoothers = {}

    start_time = time.time()
    with pose_module.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_estimator:
        drawing_utils = mp.solutions.drawing_utils
        drawing_spec_points = drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3)
        drawing_spec_lines = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=1)
        while True:
            loop_start = time.time()
            ok, frame_bgr = capture.read()
            if not ok:
                break

            elapsed = time.time() - start_time
            if elapsed >= args.run_seconds:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = pose_estimator.process(frame_rgb)

            if results.pose_landmarks:
                drawing_utils.draw_landmarks(
                    frame_bgr,
                    results.pose_landmarks,
                    pose_module.POSE_CONNECTIONS,
                    landmark_drawing_spec=drawing_spec_points,
                    connection_drawing_spec=drawing_spec_lines,
                )

            frame_features = extract_frame_features(results, landmark_indices, angle_triplets)

            if frame_features is not None:
                window.append(frame_features)
                if len(window) > args.sequence_length:
                    window.pop(0)

            normalized_label = normalize_exercise_name(current_label)
            if results.pose_landmarks and normalized_label in EXERCISE_CONFIGS:
                if normalized_label not in rep_counters:
                    config = EXERCISE_CONFIGS[normalized_label]
                    rep_counters[normalized_label] = FixedThresholdFSMCounter(config.fixed_low, config.fixed_high, config.min_state_frames)
                    rep_smoothers[normalized_label] = SmoothingBuffer(config.smoothing_window)
                landmarks = {}
                for name, index in landmark_indices.items():
                    lm = results.pose_landmarks.landmark[index]
                    landmarks[name] = np.array([lm.x, lm.y, lm.z], dtype=np.float32) if lm.visibility >= 0.5 else np.array([0.0, 0.0, 0.0], dtype=np.float32)
                current_config = EXERCISE_CONFIGS[normalized_label]
                raw_angle = extract_primary_angle(landmarks, current_config)
                smoothed_angle = rep_smoothers[normalized_label].update(raw_angle)
                rep_counters[normalized_label].update(smoothed_angle)

            if len(window) == args.sequence_length and (time.time() - last_prediction_time) >= args.prediction_interval:
                infer_start = time.time()
                sequence_array = np.array(window, dtype=np.float32).reshape(1, -1)
                scaled = scaler.transform(sequence_array).reshape(1, args.sequence_length, args.feature_count)
                input_tensor = torch.tensor(scaled, dtype=torch.float32, device=device)

                with torch.inference_mode():
                    logits = model(input_tensor)
                    prediction_index = int(torch.argmax(logits, dim=1).item())
                    current_label = label_encoder.classes_[prediction_index]

                infer_ms = (time.time() - infer_start) * 1000.0
                prediction_latencies_ms.append(infer_ms)
                predicted_labels.append(current_label)
                events.append({"timestamp_sec": elapsed, "predicted_label": current_label, "latency_ms": infer_ms})
                last_prediction_time = time.time()

            frame_times.append(time.time() - loop_start)

            cv2.putText(frame_bgr, f"Model: {args.model_name}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame_bgr, f"Pred: {current_label}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            current_reps = rep_counters[normalized_label].reps if normalized_label in rep_counters else 0
            cv2.putText(frame_bgr, f"Reps: {current_reps}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame_bgr, f"Time: {elapsed:5.1f}s", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("Realtime Evaluation", frame_bgr)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    capture.release()
    cv2.destroyAllWindows()

    fps = 1.0 / np.mean(frame_times) if frame_times else 0.0
    mean_latency = float(np.mean(prediction_latencies_ms)) if prediction_latencies_ms else None
    p95_latency = float(np.percentile(prediction_latencies_ms, 95)) if prediction_latencies_ms else None

    flips = 0
    for index in range(1, len(predicted_labels)):
        if predicted_labels[index] != predicted_labels[index - 1]:
            flips += 1
    flip_rate = float(flips / max(1, len(predicted_labels) - 1))

    summary = {
        "model": args.model_name,
        "device": str(device),
        "run_seconds": args.run_seconds,
        "prediction_count": len(predicted_labels),
        "mean_latency_ms": mean_latency,
        "p95_latency_ms": p95_latency,
        "pipeline_fps": float(fps),
        "prediction_flip_rate": flip_rate,
    }

    summary_path = output_dir / f"{args.model_name}_realtime_metrics.csv"
    events_path = output_dir / f"{args.model_name}_realtime_events.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    pd.DataFrame(events).to_csv(events_path, index=False)

    print("Realtime evaluation completed.")
    print(summary)
    print(f"Saved: {summary_path}")
    print(f"Saved: {events_path}")


if __name__ == "__main__":
    main()
