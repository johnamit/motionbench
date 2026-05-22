import argparse
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate.rep_counting_methods import EXERCISE_CONFIGS, FixedThresholdFSMCounter, SmoothingBuffer, extract_primary_angle, normalize_exercise_name


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-file", required=True)
    parser.add_argument("--output-dir", default="results/eval_rep_counting")
    parser.add_argument("--min-visibility", type=float, default=0.5)
    return parser.parse_args()


def load_pose_module():
    return mp.solutions.pose


def build_landmark_indices(mp_pose):
    names = [
        "LEFT_SHOULDER",
        "RIGHT_SHOULDER",
        "LEFT_HIP",
        "RIGHT_HIP",
        "LEFT_KNEE",
        "RIGHT_KNEE",
        "LEFT_ELBOW",
        "RIGHT_ELBOW",
        "LEFT_WRIST",
        "RIGHT_WRIST",
        "LEFT_ANKLE",
        "RIGHT_ANKLE",
    ]
    return {name: mp_pose.PoseLandmark[name].value for name in names}


def extract_landmark_points(results, landmark_indices, min_visibility):
    if not results.pose_landmarks:
        return None
    points = {}
    for name, index in landmark_indices.items():
        landmark = results.pose_landmarks.landmark[index]
        if landmark.visibility >= min_visibility:
            points[name] = np.array([landmark.x, landmark.y, landmark.z], dtype=np.float32)
        else:
            points[name] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    return points


def evaluate_video(video_path, exercise_label, pose_estimator, landmark_indices, min_visibility):
    config = EXERCISE_CONFIGS[exercise_label]
    fixed_counter = FixedThresholdFSMCounter(config.fixed_low, config.fixed_high, config.min_state_frames)
    smoothing = SmoothingBuffer(window_size=config.smoothing_window)

    capture = cv2.VideoCapture(str(video_path))
    processed_frames = 0
    valid_angle_frames = 0

    while capture.isOpened():
        read_ok, frame_bgr = capture.read()
        if not read_ok:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pose_result = pose_estimator.process(frame_rgb)
        points = extract_landmark_points(pose_result, landmark_indices, min_visibility)
        if points is None:
            processed_frames += 1
            continue

        raw_angle = extract_primary_angle(points, config)
        smoothed_angle = smoothing.update(raw_angle)
        if not np.isnan(smoothed_angle):
            valid_angle_frames += 1
        fixed_counter.update(smoothed_angle)
        processed_frames += 1

    capture.release()
    return {
        "paper_fsm": fixed_counter.reps,
        "processed_frames": processed_frames,
        "valid_angle_frames": valid_angle_frames,
    }


def compute_error_metrics(predicted_reps, true_reps):
    absolute_error = abs(predicted_reps - true_reps)
    relative_error_percent = (absolute_error / true_reps) * 100.0 if true_reps > 0 else 0.0
    missed_reps = max(0, true_reps - predicted_reps)
    false_reps = max(0, predicted_reps - true_reps)
    return absolute_error, relative_error_percent, missed_reps, false_reps


def validate_manifest_columns(dataframe):
    required = {"video_path", "exercise_label"}
    missing = required - set(dataframe.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Manifest is missing required columns: {missing_text}")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest_file)
    validate_manifest_columns(manifest)

    mp_pose = load_pose_module()
    landmark_indices = build_landmark_indices(mp_pose)
    per_video_rows = []

    has_true_reps = "true_reps" in manifest.columns

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose_estimator:
        for row in manifest.itertuples(index=False):
            video_path = Path(row.video_path)
            raw_exercise = str(row.exercise_label)
            exercise_label = normalize_exercise_name(raw_exercise)
            true_reps = int(row.true_reps) if has_true_reps and not pd.isna(row.true_reps) else None

            if exercise_label not in EXERCISE_CONFIGS:
                raise ValueError(f"Unsupported exercise '{raw_exercise}' from manifest row: {video_path}")
            if not video_path.exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")

            counts = evaluate_video(
                video_path=video_path,
                exercise_label=exercise_label,
                pose_estimator=pose_estimator,
                landmark_indices=landmark_indices,
                min_visibility=args.min_visibility,
            )
            pred_reps = counts["paper_fsm"]
            row_data = {
                "video_path": str(video_path),
                "exercise_label": exercise_label,
                "method": "paper_fsm",
                "predicted_reps": pred_reps,
                "processed_frames": counts["processed_frames"],
                "valid_angle_frames": counts["valid_angle_frames"],
            }
            if true_reps is not None:
                abs_err, rel_err, missed, false = compute_error_metrics(pred_reps, true_reps)
                row_data["true_reps"] = true_reps
                row_data["absolute_count_error"] = abs_err
                row_data["relative_error_percent"] = rel_err
                row_data["missed_reps"] = missed
                row_data["false_reps"] = false
            per_video_rows.append(row_data)

            print(
                f"{video_path.name} | {exercise_label} | "
                f"predicted_reps={counts['paper_fsm']}"
            )

    per_video_df = pd.DataFrame(per_video_rows)
    if "absolute_count_error" in per_video_df.columns:
        summary_df = (
            per_video_df.groupby("method", as_index=False)
            .agg(
                videos=("video_path", "count"),
                mean_absolute_count_error=("absolute_count_error", "mean"),
                mean_relative_error_percent=("relative_error_percent", "mean"),
                total_missed_reps=("missed_reps", "sum"),
                total_false_reps=("false_reps", "sum"),
            )
            .sort_values(by=["mean_absolute_count_error", "total_false_reps"], ascending=[True, True])
        )
    else:
        summary_df = per_video_df.groupby("method", as_index=False).agg(videos=("video_path", "count"))

    per_video_file = output_dir / "rep_counting_per_video.csv"
    summary_file = output_dir / "rep_counting_summary.csv"
    per_video_df.to_csv(per_video_file, index=False)
    summary_df.to_csv(summary_file, index=False)

    print("\nRep counting summary")
    print(summary_df.to_string(index=False))
    print(f"\nSaved: {per_video_file}")
    print(f"Saved: {summary_file}")


if __name__ == "__main__":
    main()
