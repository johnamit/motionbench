import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", default="data/train_sequences.csv")
    parser.add_argument("--models-root", default="models")
    parser.add_argument("--supported-models", nargs="+", default=["bilstm", "lstm", "gru", "tcn", "cnn_bilstm", "st_gcn"])
    parser.add_argument("--output-filename", default="similarity_centroids.pkl")
    return parser.parse_args()


MODEL_FILES = {
    "bilstm": {"scaler": "bidirectionallstm_scaler.pkl", "encoder": "bidirectionallstm_label_encoder.pkl"},
    "lstm": {"scaler": "lstm_scaler.pkl", "encoder": "lstm_label_encoder.pkl"},
    "tcn": {"scaler": "tcn_scaler.pkl", "encoder": "tcn_label_encoder.pkl"},
    "gru": {"scaler": "gru_scaler.pkl", "encoder": "gru_label_encoder.pkl"},
    "cnn_bilstm": {"scaler": "cnn_bilstm_scaler.pkl", "encoder": "cnn_bilstm_label_encoder.pkl"},
    "st_gcn": {"scaler": "st_gcn_scaler.pkl", "encoder": "st_gcn_label_encoder.pkl"},
}


def load_train_table(train_file_path):
    train_table = pd.read_csv(train_file_path)
    metadata_columns = {"video_id", "exercise_label", "start_frame_index", "end_frame_index"}
    feature_columns = [column_name for column_name in train_table.columns if column_name not in metadata_columns]
    feature_matrix = train_table[feature_columns].to_numpy(dtype=np.float32)
    label_array = train_table["exercise_label"].to_numpy()
    return feature_matrix, label_array


def compute_normalized_centroid(vectors):
    centroid = np.mean(vectors, axis=0)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm == 0.0:
        return centroid
    return centroid / centroid_norm


def build_model_centroids(model_name, model_files, models_root_path, train_features, train_labels):
    weights_dir = models_root_path / model_name / "weights"
    scaler = joblib.load(weights_dir / model_files["scaler"])
    label_encoder = joblib.load(weights_dir / model_files["encoder"])

    scaled_features = scaler.transform(train_features)
    class_names = list(label_encoder.classes_)

    centroids = {}
    sample_counts = {}
    for class_name in class_names:
        class_mask = train_labels == class_name
        class_vectors = scaled_features[class_mask]
        class_vectors = class_vectors / np.clip(np.linalg.norm(class_vectors, axis=1, keepdims=True), 1e-8, None)
        centroids[class_name] = compute_normalized_centroid(class_vectors)
        sample_counts[class_name] = int(class_vectors.shape[0])

    return {
        "model": model_name,
        "similarity_method": "cosine",
        "class_order": class_names,
        "sample_counts": sample_counts,
        "centroids": centroids,
    }


def save_model_asset(weights_dir, output_filename, similarity_asset):
    output_path = weights_dir / output_filename
    joblib.dump(similarity_asset, output_path)
    return output_path


def save_supported_models_manifest(models_root_path, supported_models):
    manifest_path = models_root_path / "similarity_supported_models.csv"
    pd.DataFrame([{"model": model_name, "method": "cosine", "version": "v1_centroid"} for model_name in supported_models]).to_csv(manifest_path, index=False)
    return manifest_path


def main():
    args = parse_args()

    train_file_path = Path(args.train_file)
    models_root_path = Path(args.models_root)
    supported_models = args.supported_models
    output_filename = args.output_filename

    train_features, train_labels = load_train_table(train_file_path)

    for model_name in supported_models:
        if model_name not in MODEL_FILES:
            print(f"Skipping unsupported model key: {model_name}")
            continue

        model_files = MODEL_FILES[model_name]
        similarity_asset = build_model_centroids(
            model_name=model_name,
            model_files=model_files,
            models_root_path=models_root_path,
            train_features=train_features,
            train_labels=train_labels,
        )

        weights_dir = models_root_path / model_name / "weights"
        output_path = save_model_asset(weights_dir, output_filename, similarity_asset)
        print(f"Saved: {output_path}")

    manifest_path = save_supported_models_manifest(models_root_path, supported_models)
    print(f"Saved: {manifest_path}")


if __name__ == "__main__":
    main()
