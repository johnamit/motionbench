import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", default="data/train_sequences.csv")
    parser.add_argument("--val-file", default="data/val_sequences.csv")
    parser.add_argument("--test-file", default="data/test_internal_sequences.csv")
    parser.add_argument("--output-dir", default="models/cnn_bilstm/results")
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--feature-count", type=int, default=78)
    parser.add_argument("--cnn-filters", type=int, default=128)
    parser.add_argument("--cnn-kernel-size", type=int, default=3)
    parser.add_argument("--lstm-units", type=int, default=73)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--batch-size", type=int, default=54)
    parser.add_argument("--epochs", type=int, default=73)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--lr-plateau-patience", type=int, default=5)
    parser.add_argument("--lr-plateau-factor", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class SequenceDataset(Dataset):
    def __init__(self, feature_tensor, label_tensor):
        self.feature_tensor = feature_tensor
        self.label_tensor = label_tensor

    def __len__(self):
        return len(self.label_tensor)

    def __getitem__(self, index):
        return self.feature_tensor[index], self.label_tensor[index]


class CnnBiLstmClassifier(nn.Module):
    def __init__(self, feature_count, class_count, cnn_filters, cnn_kernel_size, lstm_units, dropout_probability):
        super().__init__()
        cnn_padding = cnn_kernel_size // 2
        self.conv1d = nn.Conv1d(in_channels=feature_count, out_channels=cnn_filters, kernel_size=cnn_kernel_size, padding=cnn_padding)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_probability)
        self.bilstm = nn.LSTM(input_size=cnn_filters, hidden_size=lstm_units, num_layers=2, batch_first=True, dropout=dropout_probability, bidirectional=True)
        self.dropout2 = nn.Dropout(dropout_probability)
        self.classifier = nn.Linear(lstm_units * 2, class_count)

    def forward(self, input_sequence):
        temporal_tensor = input_sequence.transpose(1, 2)
        temporal_tensor = self.conv1d(temporal_tensor)
        temporal_tensor = self.relu(temporal_tensor)
        temporal_tensor = self.dropout1(temporal_tensor)
        temporal_tensor = temporal_tensor.transpose(1, 2)

        recurrent_output, _ = self.bilstm(temporal_tensor)
        final_timestep_output = recurrent_output[:, -1, :]
        dropout_output = self.dropout2(final_timestep_output)
        logits = self.classifier(dropout_output)
        return logits


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_sequence_table(input_file_path):
    sequence_table = pd.read_csv(input_file_path)
    metadata_columns = {"video_id", "exercise_label", "start_frame_index", "end_frame_index"}
    flattened_feature_columns = [column_name for column_name in sequence_table.columns if column_name not in metadata_columns]
    flattened_features = sequence_table[flattened_feature_columns].to_numpy(dtype=np.float32)
    raw_labels = sequence_table["exercise_label"].to_numpy()
    return flattened_features, raw_labels


def scale_and_reshape_features(train_features, validation_features, test_features, sequence_length, feature_count):
    scaler = StandardScaler()
    scaler.fit(train_features)

    scaled_train = scaler.transform(train_features).reshape(-1, sequence_length, feature_count)
    scaled_validation = scaler.transform(validation_features).reshape(-1, sequence_length, feature_count)
    scaled_test = scaler.transform(test_features).reshape(-1, sequence_length, feature_count)

    return scaled_train, scaled_validation, scaled_test, scaler


def build_dataloaders(train_features, validation_features, test_features, train_labels, validation_labels, test_labels, batch_size, num_workers):
    train_feature_tensor = torch.tensor(train_features, dtype=torch.float32)
    validation_feature_tensor = torch.tensor(validation_features, dtype=torch.float32)
    test_feature_tensor = torch.tensor(test_features, dtype=torch.float32)

    train_label_tensor = torch.tensor(train_labels, dtype=torch.long)
    validation_label_tensor = torch.tensor(validation_labels, dtype=torch.long)
    test_label_tensor = torch.tensor(test_labels, dtype=torch.long)

    train_dataset = SequenceDataset(train_feature_tensor, train_label_tensor)
    validation_dataset = SequenceDataset(validation_feature_tensor, validation_label_tensor)
    test_dataset = SequenceDataset(test_feature_tensor, test_label_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, validation_loader, test_loader


def run_training_epoch(model, data_loader, optimizer, loss_function, device):
    model.train()
    cumulative_loss = 0.0

    for feature_batch, label_batch in data_loader:
        feature_batch = feature_batch.to(device, non_blocking=True)
        label_batch = label_batch.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(feature_batch)
        loss = loss_function(logits, label_batch)
        loss.backward()
        optimizer.step()

        cumulative_loss += loss.item() * feature_batch.size(0)

    epoch_loss = cumulative_loss / len(data_loader.dataset)
    return epoch_loss


def run_validation_epoch(model, data_loader, loss_function, device):
    model.eval()
    cumulative_loss = 0.0

    with torch.inference_mode():
        for feature_batch, label_batch in data_loader:
            feature_batch = feature_batch.to(device, non_blocking=True)
            label_batch = label_batch.to(device, non_blocking=True)
            logits = model(feature_batch)
            loss = loss_function(logits, label_batch)
            cumulative_loss += loss.item() * feature_batch.size(0)

    epoch_loss = cumulative_loss / len(data_loader.dataset)
    return epoch_loss


def predict_labels(model, data_loader, device):
    model.eval()
    predicted_labels = []
    true_labels = []

    with torch.inference_mode():
        for feature_batch, label_batch in data_loader:
            feature_batch = feature_batch.to(device, non_blocking=True)
            logits = model(feature_batch)
            predicted_batch = torch.argmax(logits, dim=1)
            predicted_labels.append(predicted_batch.cpu().numpy())
            true_labels.append(label_batch.numpy())

    predicted_labels = np.concatenate(predicted_labels)
    true_labels = np.concatenate(true_labels)
    return true_labels, predicted_labels


def save_confusion_matrix_figure(confusion_matrix_array, class_names, output_file_path):
    figure = plt.figure(figsize=(8, 6))
    axis = figure.add_subplot(111)
    image = axis.imshow(confusion_matrix_array, interpolation="nearest", cmap="Blues")
    axis.figure.colorbar(image, ax=axis)
    axis.set_xticks(np.arange(len(class_names)))
    axis.set_yticks(np.arange(len(class_names)))
    axis.set_xticklabels(class_names, rotation=45, ha="right")
    axis.set_yticklabels(class_names)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title("Test Confusion Matrix")

    threshold = confusion_matrix_array.max() / 2.0 if confusion_matrix_array.size > 0 else 0.0
    for row_index in range(confusion_matrix_array.shape[0]):
        for column_index in range(confusion_matrix_array.shape[1]):
            value = confusion_matrix_array[row_index, column_index]
            color = "white" if value > threshold else "black"
            axis.text(column_index, row_index, str(value), ha="center", va="center", color=color)

    figure.tight_layout()
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_file_path, dpi=180)
    plt.close(figure)


def main():
    args = parse_args()

    train_file_path = Path(args.train_file)
    validation_file_path = Path(args.val_file)
    test_file_path = Path(args.test_file)
    output_directory_path = Path(args.output_dir)
    output_directory_path.mkdir(parents=True, exist_ok=True)

    sequence_length = args.sequence_length
    feature_count = args.feature_count
    cnn_filters = args.cnn_filters
    cnn_kernel_size = args.cnn_kernel_size
    lstm_units = args.lstm_units
    dropout_probability = args.dropout
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    maximum_epochs = args.epochs
    early_stopping_patience = args.early_stopping_patience
    lr_plateau_patience = args.lr_plateau_patience
    lr_plateau_factor = args.lr_plateau_factor
    num_workers = args.num_workers
    seed = args.seed

    set_random_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_features, train_raw_labels = load_sequence_table(train_file_path)
    validation_features, validation_raw_labels = load_sequence_table(validation_file_path)
    test_features, test_raw_labels = load_sequence_table(test_file_path)

    label_encoder = LabelEncoder()
    label_encoder.fit(train_raw_labels)
    train_labels = label_encoder.transform(train_raw_labels)
    validation_labels = label_encoder.transform(validation_raw_labels)
    test_labels = label_encoder.transform(test_raw_labels)

    scaled_train, scaled_validation, scaled_test, scaler = scale_and_reshape_features(
        train_features=train_features,
        validation_features=validation_features,
        test_features=test_features,
        sequence_length=sequence_length,
        feature_count=feature_count,
    )

    train_loader, validation_loader, test_loader = build_dataloaders(
        train_features=scaled_train,
        validation_features=scaled_validation,
        test_features=scaled_test,
        train_labels=train_labels,
        validation_labels=validation_labels,
        test_labels=test_labels,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    class_count = len(label_encoder.classes_)
    model = CnnBiLstmClassifier(feature_count, class_count, cnn_filters, cnn_kernel_size, lstm_units, dropout_probability).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=lr_plateau_factor, patience=lr_plateau_patience)
    loss_function = nn.CrossEntropyLoss()

    training_losses = []
    validation_losses = []
    best_validation_loss = float("inf")
    best_model_state = None
    epochs_without_improvement = 0

    for epoch_index in range(maximum_epochs):
        training_loss = run_training_epoch(model, train_loader, optimizer, loss_function, device)
        validation_loss = run_validation_epoch(model, validation_loader, loss_function, device)
        scheduler.step(validation_loss)

        training_losses.append(training_loss)
        validation_losses.append(validation_loss)

        print(f"Epoch {epoch_index + 1}/{maximum_epochs} - train_loss: {training_loss:.6f} - val_loss: {validation_loss:.6f}")

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_model_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            print("Early stopping triggered.")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    test_true_labels, test_predicted_labels = predict_labels(model, test_loader, device)

    accuracy = accuracy_score(test_true_labels, test_predicted_labels)
    precision = precision_score(test_true_labels, test_predicted_labels, average="weighted", zero_division=0)
    recall = recall_score(test_true_labels, test_predicted_labels, average="weighted", zero_division=0)
    f1 = f1_score(test_true_labels, test_predicted_labels, average="weighted", zero_division=0)
    report_text = classification_report(test_true_labels, test_predicted_labels, target_names=label_encoder.classes_, zero_division=0)
    matrix = confusion_matrix(test_true_labels, test_predicted_labels)

    print("\nTest metrics")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("\nClassification report")
    print(report_text)

    torch.save(model.state_dict(), output_directory_path / "cnn_bilstm_model.pt")
    joblib.dump(scaler, output_directory_path / "cnn_bilstm_scaler.pkl")
    joblib.dump(label_encoder, output_directory_path / "cnn_bilstm_label_encoder.pkl")

    training_history = {"training_loss": training_losses, "validation_loss": validation_losses}
    metrics = {
        "accuracy": float(accuracy),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "classes": list(label_encoder.classes_),
        "classification_report_text": report_text,
        "confusion_matrix": matrix.tolist(),
    }

    pd.DataFrame({"training_loss": training_losses, "validation_loss": validation_losses}).to_csv(output_directory_path / "training_history.csv", index=False)
    pd.DataFrame([{"accuracy": float(accuracy), "precision_weighted": float(precision), "recall_weighted": float(recall), "f1_weighted": float(f1)}]).to_csv(output_directory_path / "test_metrics.csv", index=False)
    pd.DataFrame(matrix).to_csv(output_directory_path / "test_confusion_matrix_values.csv", index=False)

    save_confusion_matrix_figure(matrix, label_encoder.classes_, output_directory_path / "test_confusion_matrix.png")
    print(f"Saved artifacts to: {output_directory_path}")


if __name__ == "__main__":
    main()
