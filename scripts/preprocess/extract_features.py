import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="data/raw/real-time-exercise-recognition-dataset")
    parser.add_argument("--input-datasets", nargs="+", default=["final_kaggle_with_additional_video", "synthetic_dataset/synthetic_dataset", "similar_dataset"])
    parser.add_argument("--output-dir", default="data/interim")
    parser.add_argument("--min-visibility", type=float, default=0.5)
    return parser.parse_args()

# Build a mapping of landmark names to their corresponding MediaPipe indices
def build_landmark_indices(mp_pose):
    return {
        "LEFT_SHOULDER": mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        "RIGHT_SHOULDER": mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        "LEFT_HIP": mp_pose.PoseLandmark.LEFT_HIP.value,
        "RIGHT_HIP": mp_pose.PoseLandmark.RIGHT_HIP.value,
        "LEFT_KNEE": mp_pose.PoseLandmark.LEFT_KNEE.value,
        "RIGHT_KNEE": mp_pose.PoseLandmark.RIGHT_KNEE.value,
        "LEFT_ELBOW": mp_pose.PoseLandmark.LEFT_ELBOW.value,
        "RIGHT_ELBOW": mp_pose.PoseLandmark.RIGHT_ELBOW.value,
        "LEFT_WRIST": mp_pose.PoseLandmark.LEFT_WRIST.value,
        "RIGHT_WRIST": mp_pose.PoseLandmark.RIGHT_WRIST.value,
        "LEFT_ANKLE": mp_pose.PoseLandmark.LEFT_ANKLE.value,
        "RIGHT_ANKLE": mp_pose.PoseLandmark.RIGHT_ANKLE.value,
        "LEFT_HEEL": mp_pose.PoseLandmark.LEFT_HEEL.value,
        "RIGHT_HEEL": mp_pose.PoseLandmark.RIGHT_HEEL.value,
        "LEFT_FOOT_INDEX": mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value,
        "RIGHT_FOOT_INDEX": mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value,
        "LEFT_PINKY": mp_pose.PoseLandmark.LEFT_PINKY.value,
        "RIGHT_PINKY": mp_pose.PoseLandmark.RIGHT_PINKY.value,
        "LEFT_INDEX": mp_pose.PoseLandmark.LEFT_INDEX.value,
        "RIGHT_INDEX": mp_pose.PoseLandmark.RIGHT_INDEX.value,
        "LEFT_THUMB": mp_pose.PoseLandmark.LEFT_THUMB.value,
        "RIGHT_THUMB": mp_pose.PoseLandmark.RIGHT_THUMB.value,
    }

# Define the list of landmark names we want to extract coordinates for
def get_coordinate_landmark_names():
    return [
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
        "LEFT_HEEL",
        "RIGHT_HEEL",
        "LEFT_FOOT_INDEX",
        "RIGHT_FOOT_INDEX",
        "LEFT_PINKY",
        "RIGHT_PINKY",
        "LEFT_INDEX",
        "RIGHT_INDEX",
        "LEFT_THUMB",
        "RIGHT_THUMB",
    ]

# Define the triplets of landmarks for which we want to calculate joint angles
def get_angle_triplets():
    return [
        ("LEFT_HIP", "LEFT_SHOULDER", "LEFT_ELBOW"),
        ("RIGHT_HIP", "RIGHT_SHOULDER", "RIGHT_ELBOW"),
        ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
        ("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),
        ("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"),
        ("RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"),
        ("LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE"),
        ("RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE"),
        ("LEFT_KNEE", "LEFT_ANKLE", "LEFT_HEEL"),
        ("RIGHT_KNEE", "RIGHT_ANKLE", "RIGHT_HEEL"),
        ("LEFT_ANKLE", "LEFT_HEEL", "LEFT_FOOT_INDEX"),
        ("RIGHT_ANKLE", "RIGHT_HEEL", "RIGHT_FOOT_INDEX"),
    ]

# function to recursively list video files in a dataset folder, filtering by video file extensions
def list_video_files(dataset_path):
    allowed_suffixes = {".mp4", ".avi", ".mov", ".m4v", ".asf", ".MOV"}
    video_paths = []
    for file_path in dataset_path.rglob("*"):
        if file_path.is_file() and file_path.suffix in allowed_suffixes:
            video_paths.append(file_path)
    return sorted(video_paths)

# normalise exercise labels by mapping known variations to a standard label, and lowercasing/stripping whitespace for consistency
def normalize_exercise_label(raw_label):
    lower_label = raw_label.strip().lower()
    label_mapping = {
        "hammer curl": "barbell biceps curl",
        "bicept curl": "barbell biceps curl",
    }
    return label_mapping.get(lower_label, lower_label)

# Check if a MediaPipe landmark is valid based on its visibility score compared to a minimum threshold
def is_landmark_valid(landmark, min_visibility):
    return landmark.visibility >= min_visibility

# Return a placeholder point (0, 0, 0) for missing or invalid landmarks to maintain consistent feature dimensions
def get_placeholder_point():
    return np.array([0.0, 0.0, 0.0], dtype=np.float32)

# Calculate the angle in degrees between three points (A, B, C) where B is the vertex point. If any point is invalid (all zeros), return 0 degrees.
def calculate_angle_degrees(point_a, point_b, point_c):
    if np.allclose(point_a, 0.0) or np.allclose(point_b, 0.0) or np.allclose(point_c, 0.0):
        return 0.0

    vector_ab = point_a[:2] - point_b[:2]
    vector_cb = point_c[:2] - point_b[:2]
    denominator = np.linalg.norm(vector_ab) * np.linalg.norm(vector_cb)

    if denominator == 0.0:
        return 0.0

    cosine_value = np.dot(vector_ab, vector_cb) / denominator 
    cosine_value = np.clip(cosine_value, -1.0, 1.0) # Clip cosine value to the valid range to avoid numerical issues with arccos
    angle_radians = np.arccos(cosine_value) # Calculate angle in radians and convert to degrees
    return float(np.degrees(angle_radians))

# Extract the specified landmarks from a MediaPipe pose estimation result for a single frame, checking visibility and using placeholders for missing landmarks. Returns a dictionary of landmark names to their (x, y, z) coordinates.
def extract_frame_landmarks(media_pipe_results, landmark_indices, coordinate_landmark_names, min_visibility):
    extracted_landmarks = {}
    if not media_pipe_results.pose_landmarks:
        return extracted_landmarks

    for landmark_name in coordinate_landmark_names:
        landmark_index = landmark_indices[landmark_name]
        detected_landmark = media_pipe_results.pose_landmarks.landmark[landmark_index]

        if is_landmark_valid(detected_landmark, min_visibility):
            extracted_landmarks[landmark_name] = np.array(
                [detected_landmark.x, detected_landmark.y, detected_landmark.z],
                dtype=np.float32,
            )
        else:
            extracted_landmarks[landmark_name] = get_placeholder_point()

    return extracted_landmarks

# Check if all essential landmarks for a given body side (LEFT or RIGHT) are valid (not all zeros) to determine if we can trust the pose estimation for that side. This helps filter out frames where the pose estimation failed for one side of the body.
def has_valid_body_side(extracted_landmarks, side_prefix):
    essential_points = ["SHOULDER", "ELBOW", "WRIST", "HIP", "KNEE", "ANKLE"]
    for point_name in essential_points:
        full_landmark_name = f"{side_prefix}_{point_name}"
        if np.allclose(extracted_landmarks[full_landmark_name], 0.0):
            return False
    return True

# Build a feature row dictionary for a single frame, including the video identifier, exercise label, frame index, landmark coordinates, and calculated angles based on the specified triplets. This function combines all the extracted information into a structured format for later saving to CSV.
def build_feature_row(
    extracted_landmarks,
    coordinate_landmark_names,
    angle_triplets,
    frame_index,
    video_identifier,
    exercise_label,
):
    feature_row = {
        "video_id": video_identifier,
        "exercise_label": exercise_label,
        "frame_index": frame_index,
    }

    for landmark_name in coordinate_landmark_names:
        landmark_value = extracted_landmarks[landmark_name]
        feature_row[f"{landmark_name.lower()}_x"] = landmark_value[0]
        feature_row[f"{landmark_name.lower()}_y"] = landmark_value[1]
        feature_row[f"{landmark_name.lower()}_z"] = landmark_value[2]

    for point_a, point_b, point_c in angle_triplets:
        angle_name = f"angle_{point_a.lower()}_{point_b.lower()}_{point_c.lower()}"
        feature_row[angle_name] = calculate_angle_degrees(
            extracted_landmarks[point_a],
            extracted_landmarks[point_b],
            extracted_landmarks[point_c],
        )

    return feature_row

# Process a single video file to extract per-frame features. For each frame, it runs pose estimation, extracts landmarks, checks validity, and builds feature rows for valid frames. It returns a list of feature rows for the entire video.
def extract_features_from_video(
    video_path,
    exercise_label,
    dataset_name,
    pose_estimator,
    landmark_indices,
    coordinate_landmark_names,
    angle_triplets,
    min_visibility,
):
    video_capture = cv2.VideoCapture(str(video_path))
    frame_index = 0
    extracted_rows = []
    video_identifier = f"{dataset_name}/{exercise_label}/{video_path.stem}"

    while video_capture.isOpened():
        frame_read_success, frame_bgr = video_capture.read()
        if not frame_read_success:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pose_results = pose_estimator.process(frame_rgb)
        extracted_landmarks = extract_frame_landmarks(
            pose_results,
            landmark_indices,
            coordinate_landmark_names,
            min_visibility,
        )

        if extracted_landmarks:
            left_side_is_valid = has_valid_body_side(extracted_landmarks, "LEFT")
            right_side_is_valid = has_valid_body_side(extracted_landmarks, "RIGHT")

            if left_side_is_valid or right_side_is_valid:
                extracted_rows.append(
                    build_feature_row(
                        extracted_landmarks=extracted_landmarks,
                        coordinate_landmark_names=coordinate_landmark_names,
                        angle_triplets=angle_triplets,
                        frame_index=frame_index,
                        video_identifier=video_identifier,
                        exercise_label=exercise_label,
                    )
                )

        frame_index += 1

    video_capture.release()
    return extracted_rows

# Save the list of feature row dictionaries to a CSV file using pandas, ensuring the output directory exists. Each row in the CSV corresponds to a single frame's extracted features.
def save_rows_to_csv(rows, output_file_path):
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(output_file_path, index=False)

# Sets up the main execution flow: parses arguments, initializes MediaPipe pose estimator, iterates through specified datasets and videos, extracts features for each video, and saves the results to CSV files in the output directory.
def main():
    args = parse_args()

    dataset_root_path = Path(args.dataset_root)
    input_dataset_names = args.input_datasets
    output_directory_path = Path(args.output_dir)
    minimum_landmark_visibility = args.min_visibility

    media_pipe_pose = mp.solutions.pose
    landmark_indices = build_landmark_indices(media_pipe_pose)
    coordinate_landmark_names = get_coordinate_landmark_names()
    angle_triplets = get_angle_triplets()

    with media_pipe_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose_estimator:
        for dataset_name in input_dataset_names:
            dataset_path = dataset_root_path / dataset_name
            if not dataset_path.exists():
                print(f"Skipping missing dataset folder: {dataset_path}")
                continue

            print(f"Processing dataset: {dataset_name}")
            dataset_rows = []
            video_paths = list_video_files(dataset_path)

            for video_path in tqdm(video_paths, desc=f"Videos in {dataset_name}"):
                exercise_label = normalize_exercise_label(video_path.parent.name)
                video_rows = extract_features_from_video(
                    video_path=video_path,
                    exercise_label=exercise_label,
                    dataset_name=dataset_name,
                    pose_estimator=pose_estimator,
                    landmark_indices=landmark_indices,
                    coordinate_landmark_names=coordinate_landmark_names,
                    angle_triplets=angle_triplets,
                    min_visibility=minimum_landmark_visibility,
                )
                dataset_rows.extend(video_rows)

            output_file_path = output_directory_path / f"{dataset_name.replace('/', '_')}_frame_features.csv"
            save_rows_to_csv(dataset_rows, output_file_path)
            print(f"Saved: {output_file_path} ({len(dataset_rows)} rows)")


if __name__ == "__main__":
    main()
