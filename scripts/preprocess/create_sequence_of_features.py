import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/interim")
    parser.add_argument("--input-pattern", default="*_frame_features.csv")
    parser.add_argument("--output-file", default="data/train_sequences_full.csv")
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--stride", type=int, default=30)
    return parser.parse_args()

# Load all per-frame feature tables from the specified directory and pattern
def load_frame_feature_tables(input_directory_path, input_pattern):
    frame_feature_paths = sorted(input_directory_path.glob(input_pattern))
    if not frame_feature_paths:
        raise FileNotFoundError(f"No files found in {input_directory_path} matching {input_pattern}")

    dataframes = []
    for frame_feature_path in frame_feature_paths:
        dataframe = pd.read_csv(frame_feature_path)
        dataframes.append(dataframe)
    return dataframes

# Extract the names of the feature columns, excluding metadata columns
def get_frame_feature_column_names(frame_feature_table):
    excluded_columns = {"video_id", "exercise_label", "frame_index"}
    frame_feature_column_names = [
        column_name for column_name in frame_feature_table.columns if column_name not in excluded_columns
    ]
    return frame_feature_column_names

# Build a list of flattened feature names for the sequence table based on the frame feature column names and sequence length
def build_flattened_sequence_feature_names(sequence_length, frame_feature_column_names):
    flattened_feature_names = []
    for timestep_index in range(sequence_length):
        for feature_name in frame_feature_column_names:
            flattened_feature_names.append(f"t{timestep_index:02d}_{feature_name}")
    return flattened_feature_names

# Create fixed-length sequences of frame features for a single video, returning a list of sequence rows with metadata and flattened features
def create_sequences_from_video_table(video_table, frame_feature_column_names, sequence_length, stride):
    sequence_rows = []
    sorted_video_table = video_table.sort_values("frame_index")
    total_frames = len(sorted_video_table)
    max_start_index = total_frames - sequence_length

    if max_start_index < 0:
        return sequence_rows

    for start_index in range(0, max_start_index + 1, stride):
        end_index = start_index + sequence_length
        sequence_slice = sorted_video_table.iloc[start_index:end_index]
        sequence_label = sequence_slice.iloc[0]["exercise_label"]
        sequence_video_id = sequence_slice.iloc[0]["video_id"]
        sequence_feature_matrix = sequence_slice[frame_feature_column_names].to_numpy(dtype=np.float32)

        sequence_rows.append(
            {
                "video_id": sequence_video_id,
                "exercise_label": sequence_label,
                "start_frame_index": int(sequence_slice.iloc[0]["frame_index"]),
                "end_frame_index": int(sequence_slice.iloc[-1]["frame_index"]),
                "flattened_features": sequence_feature_matrix.reshape(-1),
            }
        )

    return sequence_rows

# Convert the list of per-frame feature tables into a single sequence table with flattened features for each sequence, including metadata columns for video ID, exercise label, and frame indices
def convert_frame_tables_to_sequence_table(frame_feature_tables, sequence_length, stride):
    merged_frame_feature_table = pd.concat(frame_feature_tables, ignore_index=True)
    frame_feature_column_names = get_frame_feature_column_names(merged_frame_feature_table)
    flattened_feature_names = build_flattened_sequence_feature_names(sequence_length, frame_feature_column_names)

    all_sequence_rows = []
    grouped_video_tables = merged_frame_feature_table.groupby("video_id", sort=False)

    for _, video_table in grouped_video_tables:
        video_sequences = create_sequences_from_video_table(
            video_table=video_table,
            frame_feature_column_names=frame_feature_column_names,
            sequence_length=sequence_length,
            stride=stride,
        )
        all_sequence_rows.extend(video_sequences)

    sequence_table_rows = []
    for sequence_row in all_sequence_rows:
        flat_feature_values = sequence_row["flattened_features"]
        flattened_feature_dict = dict(zip(flattened_feature_names, flat_feature_values))

        output_row = {
            "video_id": sequence_row["video_id"],
            "exercise_label": sequence_row["exercise_label"],
            "start_frame_index": sequence_row["start_frame_index"],
            "end_frame_index": sequence_row["end_frame_index"],
        }
        output_row.update(flattened_feature_dict)
        sequence_table_rows.append(output_row)

    return pd.DataFrame(sequence_table_rows)

# Save the final sequence table to a CSV file
def save_sequence_table(sequence_table, output_file_path):
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    sequence_table.to_csv(output_file_path, index=False)

# Loads per-frame feature tables, converts them into fixed-length sequences with flattened features, and saves the resulting sequence table to a CSV file
def main():
    args = parse_args()

    input_directory_path = Path(args.input_dir)
    input_pattern = args.input_pattern
    output_file_path = Path(args.output_file)
    sequence_length = args.sequence_length
    stride = args.stride

    frame_feature_tables = load_frame_feature_tables(input_directory_path, input_pattern)
    sequence_table = convert_frame_tables_to_sequence_table(frame_feature_tables, sequence_length, stride)
    save_sequence_table(sequence_table, output_file_path)

    print(f"Saved: {output_file_path}")
    print(f"Sequences: {len(sequence_table)}")


if __name__ == "__main__":
    main()
