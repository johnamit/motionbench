from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class ExerciseAngleConfig:
    primary_triplet_left: tuple[str, str, str]
    primary_triplet_right: tuple[str, str, str]
    fixed_low: float
    fixed_high: float
    min_state_frames: int = 2
    smoothing_window: int = 5


EXERCISE_CONFIGS: Dict[str, ExerciseAngleConfig] = {
    "squat": ExerciseAngleConfig(
        primary_triplet_left=("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"),
        primary_triplet_right=("RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"),
        fixed_low=95.0,
        fixed_high=160.0,
    ),
    "push up": ExerciseAngleConfig(
        primary_triplet_left=("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
        primary_triplet_right=("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),
        fixed_low=95.0,
        fixed_high=155.0,
    ),
    "barbell biceps curl": ExerciseAngleConfig(
        primary_triplet_left=("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
        primary_triplet_right=("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),
        fixed_low=55.0,
        fixed_high=145.0,
    ),
    "shoulder press": ExerciseAngleConfig(
        primary_triplet_left=("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
        primary_triplet_right=("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),
        fixed_low=70.0,
        fixed_high=155.0,
    ),
}


EXERCISE_ALIASES = {
    "push-up": "push up",
    "pushups": "push up",
    "pushup": "push up",
    "curls": "barbell biceps curl",
    "bicep curl": "barbell biceps curl",
    "biceps curl": "barbell biceps curl",
    "shoulder_press": "shoulder press",
}


def normalize_exercise_name(exercise_name: str) -> str:
    key = exercise_name.strip().lower()
    return EXERCISE_ALIASES.get(key, key)


def calculate_angle_degrees(point_a: np.ndarray, point_b: np.ndarray, point_c: np.ndarray) -> float:
    if np.allclose(point_a, 0.0) or np.allclose(point_b, 0.0) or np.allclose(point_c, 0.0):
        return np.nan
    vector_ab = point_a[:2] - point_b[:2]
    vector_cb = point_c[:2] - point_b[:2]
    denominator = np.linalg.norm(vector_ab) * np.linalg.norm(vector_cb)
    if denominator == 0.0:
        return np.nan
    cosine_value = np.clip(np.dot(vector_ab, vector_cb) / denominator, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine_value)))


def extract_primary_angle(landmarks: Dict[str, np.ndarray], config: ExerciseAngleConfig) -> float:
    left_angle = calculate_angle_degrees(
        landmarks[config.primary_triplet_left[0]],
        landmarks[config.primary_triplet_left[1]],
        landmarks[config.primary_triplet_left[2]],
    )
    right_angle = calculate_angle_degrees(
        landmarks[config.primary_triplet_right[0]],
        landmarks[config.primary_triplet_right[1]],
        landmarks[config.primary_triplet_right[2]],
    )
    if np.isnan(left_angle) and np.isnan(right_angle):
        return np.nan
    if np.isnan(left_angle):
        return right_angle
    if np.isnan(right_angle):
        return left_angle
    return float((left_angle + right_angle) / 2.0)


class SmoothingBuffer:
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.values: List[float] = []

    def update(self, value: float) -> float:
        if np.isnan(value):
            return np.nan
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)
        return float(np.mean(self.values))


class FixedThresholdFSMCounter:
    def __init__(self, low_threshold: float, high_threshold: float, min_state_frames: int = 2):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.min_state_frames = min_state_frames
        self.reps = 0
        self.current_state = "unknown"
        self.pending_state = "unknown"
        self.pending_state_frames = 0

    def _angle_state(self, angle: float) -> str:
        if angle <= self.low_threshold:
            return "flexed"
        if angle >= self.high_threshold:
            return "extended"
        return "mid"

    def update(self, angle: float) -> int:
        if np.isnan(angle):
            return self.reps

        next_state = self._angle_state(angle)
        if next_state == "mid":
            self.pending_state = "unknown"
            self.pending_state_frames = 0
            return self.reps

        if next_state == self.pending_state:
            self.pending_state_frames += 1
        else:
            self.pending_state = next_state
            self.pending_state_frames = 1

        if self.pending_state_frames < self.min_state_frames:
            return self.reps

        if self.current_state != self.pending_state:
            previous_state = self.current_state
            self.current_state = self.pending_state
            if previous_state == "flexed" and self.current_state == "extended":
                self.reps += 1
        return self.reps
