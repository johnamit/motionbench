import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", default="data/test_home_sequences.csv")
    parser.add_argument("--models-root", default="models")
    parser.add_argument("--output-dir", default="results/benchmark_inference")
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--feature-count", type=int, default=78)
    parser.add_argument("--warmup-runs", type=int, default=50)
    parser.add_argument("--timed-runs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
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
        dropout_output = self.dropout(final_timestep_output)
        return self.classifier(dropout_output)


class LstmClassifier(nn.Module):
    def __init__(self, feature_count, hidden_size, class_count, dropout_probability):
        super().__init__()
        self.lstm = nn.LSTM(input_size=feature_count, hidden_size=hidden_size, num_layers=2, batch_first=True, dropout=dropout_probability, bidirectional=False)
        self.dropout = nn.Dropout(dropout_probability)
        self.classifier = nn.Linear(hidden_size, class_count)

    def forward(self, input_sequence):
        recurrent_output, _ = self.lstm(input_sequence)
        final_timestep_output = recurrent_output[:, -1, :]
        dropout_output = self.dropout(final_timestep_output)
        return self.classifier(dropout_output)


class GruClassifier(nn.Module):
    def __init__(self, feature_count, hidden_size, class_count, dropout_probability):
        super().__init__()
        self.gru = nn.GRU(input_size=feature_count, hidden_size=hidden_size, num_layers=2, batch_first=True, dropout=dropout_probability, bidirectional=False)
        self.dropout = nn.Dropout(dropout_probability)
        self.classifier = nn.Linear(hidden_size, class_count)

    def forward(self, input_sequence):
        recurrent_output, _ = self.gru(input_sequence)
        final_timestep_output = recurrent_output[:, -1, :]
        dropout_output = self.dropout(final_timestep_output)
        return self.classifier(dropout_output)


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
        output_tensor = self.conv1(input_tensor)
        output_tensor = self.chomp1(output_tensor)
        output_tensor = self.relu1(output_tensor)
        output_tensor = self.dropout1(output_tensor)
        output_tensor = self.conv2(output_tensor)
        output_tensor = self.chomp2(output_tensor)
        output_tensor = self.relu2(output_tensor)
        output_tensor = self.dropout2(output_tensor)
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
        temporal_tensor = self.input_projection(temporal_tensor)
        temporal_tensor = self.block1(temporal_tensor)
        temporal_tensor = self.block2(temporal_tensor)
        temporal_tensor = self.block3(temporal_tensor)
        final_timestep_tensor = temporal_tensor[:, :, -1]
        return self.classifier(final_timestep_tensor)


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
        return self.classifier(dropout_output)


class GraphConvolution(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.projection = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, input_tensor, adjacency_matrix):
        projected_tensor = self.projection(input_tensor)
        return torch.einsum("nctv,vw->nctw", projected_tensor, adjacency_matrix)


class StGcnBlock(nn.Module):
    def __init__(self, input_channels, output_channels, dropout, stride=1):
        super().__init__()
        self.graph_convolution = GraphConvolution(input_channels, output_channels)
        self.temporal_convolution = nn.Sequential(
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=(9, 1), stride=(stride, 1), padding=(4, 0)),
            nn.BatchNorm2d(output_channels),
            nn.Dropout(dropout),
        )
        if stride != 1 or input_channels != output_channels:
            self.residual = nn.Sequential(nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=(stride, 1)), nn.BatchNorm2d(output_channels))
        else:
            self.residual = nn.Identity()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, input_tensor, adjacency_matrix):
        residual_tensor = self.residual(input_tensor)
        output_tensor = self.graph_convolution(input_tensor, adjacency_matrix)
        output_tensor = self.temporal_convolution(output_tensor)
        return self.activation(output_tensor + residual_tensor)


class StGcnClassifier(nn.Module):
    def __init__(self, feature_count, class_count, dropout):
        super().__init__()
        self.input_batch_norm = nn.BatchNorm1d(feature_count)
        self.register_parameter("adjacency_logits", nn.Parameter(torch.eye(feature_count)))
        self.block1 = StGcnBlock(1, 64, dropout=dropout, stride=1)
        self.block2 = StGcnBlock(64, 64, dropout=dropout, stride=1)
        self.block3 = StGcnBlock(64, 128, dropout=dropout, stride=1)
        self.classifier = nn.Linear(128, class_count)

    def get_normalized_adjacency(self):
        return torch.softmax(self.adjacency_logits, dim=1)

    def forward(self, input_sequence):
        batch_size, sequence_length, feature_count = input_sequence.shape
        normalized_input = input_sequence.reshape(batch_size * sequence_length, feature_count)
        normalized_input = self.input_batch_norm(normalized_input)
        normalized_input = normalized_input.reshape(batch_size, sequence_length, feature_count)
        graph_tensor = normalized_input.unsqueeze(1)
        adjacency_matrix = self.get_normalized_adjacency()
        graph_tensor = self.block1(graph_tensor, adjacency_matrix)
        graph_tensor = self.block2(graph_tensor, adjacency_matrix)
        graph_tensor = self.block3(graph_tensor, adjacency_matrix)
        pooled_tensor = graph_tensor.mean(dim=2).mean(dim=2)
        return self.classifier(pooled_tensor)


MODEL_SPECS = {
    "bilstm": {
        "weight": "bidirectionallstm_model.pt",
        "scaler": "bidirectionallstm_scaler.pkl",
        "encoder": "bidirectionallstm_label_encoder.pkl",
        "builder": lambda feature_count, class_count: BidirectionalLstmClassifier(feature_count, 73, class_count, 0.2174),
    },
    "lstm": {
        "weight": "lstm_model.pt",
        "scaler": "lstm_scaler.pkl",
        "encoder": "lstm_label_encoder.pkl",
        "builder": lambda feature_count, class_count: LstmClassifier(feature_count, 117, class_count, 0.3829),
    },
    "gru": {
        "weight": "gru_model.pt",
        "scaler": "gru_scaler.pkl",
        "encoder": "gru_label_encoder.pkl",
        "builder": lambda feature_count, class_count: GruClassifier(feature_count, 96, class_count, 0.2),
    },
    "tcn": {
        "weight": "tcn_model.pt",
        "scaler": "tcn_scaler.pkl",
        "encoder": "tcn_label_encoder.pkl",
        "builder": lambda feature_count, class_count: TcnClassifier(feature_count, class_count, 128, 3, 0.2),
    },
    "cnn_bilstm": {
        "weight": "cnn_bilstm_model.pt",
        "scaler": "cnn_bilstm_scaler.pkl",
        "encoder": "cnn_bilstm_label_encoder.pkl",
        "builder": lambda feature_count, class_count: CnnBiLstmClassifier(feature_count, class_count, 128, 3, 73, 0.2),
    },
    "st_gcn": {
        "weight": "st_gcn_model.pt",
        "scaler": "st_gcn_scaler.pkl",
        "encoder": "st_gcn_label_encoder.pkl",
        "builder": lambda feature_count, class_count: StGcnClassifier(feature_count, class_count, 0.2),
    },
}


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_features(input_file_path):
    table = pd.read_csv(input_file_path)
    metadata_columns = {"video_id", "exercise_label", "start_frame_index", "end_frame_index"}
    feature_columns = [column_name for column_name in table.columns if column_name not in metadata_columns]
    return table[feature_columns].to_numpy(dtype=np.float32)


def run_device_benchmark(model, input_tensor, device, warmup_runs, timed_runs):
    model = model.to(device)
    input_tensor = input_tensor.to(device)
    model.eval()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for _ in range(warmup_runs):
            _ = model(input_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        latencies_ms = []
        for _ in range(timed_runs):
            step_start = time.perf_counter()
            _ = model(input_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            latencies_ms.append((time.perf_counter() - step_start) * 1000.0)

    mean_latency_ms = float(np.mean(latencies_ms))
    p95_latency_ms = float(np.percentile(latencies_ms, 95))

    peak_memory_mb = None
    if device.type == "cuda":
        peak_memory_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))

    return {
        "mean_latency_ms": mean_latency_ms,
        "p95_latency_ms": p95_latency_ms,
        "peak_memory_mb": peak_memory_mb,
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    input_file_path = Path(args.input_file)
    models_root_path = Path(args.models_root)
    output_directory_path = Path(args.output_dir)
    output_directory_path.mkdir(parents=True, exist_ok=True)

    full_features = load_features(input_file_path)
    single_window_features = full_features[0:1]

    benchmark_rows = []

    cpu_device = torch.device("cpu")
    has_cuda = torch.cuda.is_available()
    cuda_device = torch.device("cuda") if has_cuda else None

    for model_name, spec in MODEL_SPECS.items():
        weights_root = models_root_path / model_name / "weights"

        scaler = joblib.load(weights_root / spec["scaler"])
        label_encoder = joblib.load(weights_root / spec["encoder"])
        class_count = len(label_encoder.classes_)

        scaled_window = scaler.transform(single_window_features).reshape(1, args.sequence_length, args.feature_count)
        input_tensor = torch.tensor(scaled_window, dtype=torch.float32)

        model = spec["builder"](args.feature_count, class_count)
        state_dict = torch.load(weights_root / spec["weight"], map_location="cpu")
        model.load_state_dict(state_dict)

        model_size_mb = float((weights_root / spec["weight"]).stat().st_size / (1024.0 * 1024.0))

        cpu_stats = run_device_benchmark(model, input_tensor, cpu_device, args.warmup_runs, args.timed_runs)
        benchmark_rows.append(
            {
                "model": model_name,
                "device": "cpu",
                "model_size_mb": model_size_mb,
                **cpu_stats,
            }
        )

        if has_cuda:
            cuda_stats = run_device_benchmark(model, input_tensor, cuda_device, args.warmup_runs, args.timed_runs)
            benchmark_rows.append(
                {
                    "model": model_name,
                    "device": "cuda",
                    "model_size_mb": model_size_mb,
                    **cuda_stats,
                }
            )

        print(f"Benchmarked: {model_name}")

    benchmark_table = pd.DataFrame(benchmark_rows)

    cpu_table = benchmark_table[benchmark_table["device"] == "cpu"].sort_values("mean_latency_ms", ascending=True)
    cpu_csv_output_path = output_directory_path / "inference_benchmark_cpu.csv"
    cpu_table.to_csv(cpu_csv_output_path, index=False)

    print("\nInference Benchmark (CPU)")
    print(cpu_table.to_string(index=False))
    print(f"\nSaved: {cpu_csv_output_path}")

    if has_cuda:
        cuda_table = benchmark_table[benchmark_table["device"] == "cuda"].sort_values("mean_latency_ms", ascending=True)
        cuda_csv_output_path = output_directory_path / "inference_benchmark_cuda.csv"
        cuda_table.to_csv(cuda_csv_output_path, index=False)

        print("\nInference Benchmark (CUDA)")
        print(cuda_table.to_string(index=False))
        print(f"\nSaved: {cuda_csv_output_path}")


if __name__ == "__main__":
    main()
