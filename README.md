<img src="assets/motionbench.png" alt="MotionBench" width="900"><br>

[<img src="https://img.shields.io/badge/HuggingFace-Dataset-black?style=for-the-badge&logo=huggingface&logoColor=FFD21E&labelColor=ff7f1e" alt="View Dataset on Hugging Face"/>](YOUR_HF_DATASET_URL)
&nbsp;&nbsp;
[<img src="https://img.shields.io/badge/HuggingFace-Models-black?style=for-the-badge&logo=huggingface&logoColor=FFD21E&labelColor=ff7f1e" alt="View Models on Hugging Face"/>](YOUR_HF_MODELS_URL)

MotionBench is a real-time pose-based exercise recognition project designed for practical local usage. It classifies exercise motion from short temporal windows, estimates repetition counts with a deterministic finite-state method, and reports similarity against class-level motion centroids.

<p>
  <a href="#overview"><img src="https://img.shields.io/badge/Overview-111111?style=for-the-badge" alt="Overview"></a>
  <a href="#dataset"><img src="https://img.shields.io/badge/Dataset-111111?style=for-the-badge" alt="Dataset"></a>
  <a href="#models"><img src="https://img.shields.io/badge/Models-111111?style=for-the-badge" alt="Models"></a>
  <a href="#training"><img src="https://img.shields.io/badge/Training-111111?style=for-the-badge" alt="Training"></a>
  <a href="#inference-local"><img src="https://img.shields.io/badge/Inference-111111?style=for-the-badge" alt="Inference"></a>
  <a href="#streamlit-app"><img src="https://img.shields.io/badge/Streamlit-111111?style=for-the-badge" alt="Streamlit"></a>
  <a href="#citations"><img src="https://img.shields.io/badge/Citations-111111?style=for-the-badge" alt="Citations"></a>
  <a href="#license"><img src="https://img.shields.io/badge/License-111111?style=for-the-badge" alt="License"></a>
</p>

## Overview
MotionBench is built to run the full workflow from start to finish. You can prepare sequence data, train models, benchmark inference, and run real-time prediction from a webcam.

The runtime pipeline is simple. It captures frames, extracts pose-based features, builds rolling windows, and predicts one of six exercise classes. It also estimates repetitions with a deterministic finite-state method and reports a centroid similarity score for live feedback.

The active project layout is intentionally minimal. Core work happens in `data/`, `models/`, `scripts/`, and `results/`. Older or non-essential files are moved to `archive/` to keep the main repository clear and easy to review.

## Dataset
This repo stays lightweight on GitHub. Download the dataset files from Hugging Face and place them in `data/`.

- `YOUR_HF_DATASET_URL`

```bash
git clone YOUR_HF_DATASET_URL data
```

For local usage, keep split files under `data/`.

The workflow expects fixed sequence splits (`train`, `val`, `test_internal`) and optionally a separate home/generalization test split.

Expected split files:
- `data/train_sequences.csv`
- `data/val_sequences.csv`
- `data/test_internal_sequences.csv`
- `data/test_home_sequences.csv` (optional for home/generalization evaluation)

To regenerate centralized fixed splits:

```bash
python scripts/preprocess/create_fixed_splits.py --input-file data/train_sequences_full.csv --output-dir data
```

## Models
Download trained model files from Hugging Face and place them in `models/`.

- `YOUR_HF_MODELS_URL`

```bash
git clone YOUR_HF_MODELS_URL models
```

This project includes six sequence models with different strengths. Some are strong on temporal memory, some are better for latency, and some are better at capturing structured feature relationships.

**BiLSTM**
The bidirectional LSTM processes each sequence in forward and backward directions within the input window, so the classifier can use context from both ends of the motion segment. This is useful when discriminative movement cues are distributed across the full window rather than concentrated at one time point.

**LSTM**
The unidirectional LSTM models temporal dependencies in a simpler recurrent setup. It is often a strong baseline for motion classification and can offer a good tradeoff between model capacity and runtime cost.

**GRU**
The GRU uses gating similar to LSTM but with fewer internal components, which can reduce parameter count and improve efficiency. In practice, it is a strong candidate when you want robust sequence modeling with lighter recurrent overhead.

**TCN**
The temporal convolutional network uses dilated 1D convolutions and residual blocks to capture short- and long-range temporal structure. Because convolutional operations are parallelizable, TCNs can perform well in latency-oriented settings.

**CNN-BiLSTM**
This hybrid architecture first applies temporal convolutions to extract local motion patterns and then uses a BiLSTM to model higher-level sequence context. It combines local feature extraction with recurrent temporal integration.

**ST-GCN (feature-graph variant)**
The ST-GCN-style model treats the per-frame feature dimension as a graph-like structure and applies graph-temporal processing blocks. This can help when relationships between feature nodes are informative for classification.

## Training
Train each model from the shared sequence splits in `data/`.

```bash
python models/bilstm/train.py --train-file data/train_sequences.csv --val-file data/val_sequences.csv --test-file data/test_internal_sequences.csv --output-dir models/bilstm/results
python models/lstm/train.py --train-file data/train_sequences.csv --val-file data/val_sequences.csv --test-file data/test_internal_sequences.csv --output-dir models/lstm/results
python models/gru/train.py --train-file data/train_sequences.csv --val-file data/val_sequences.csv --test-file data/test_internal_sequences.csv --output-dir models/gru/results
python models/tcn/train.py --train-file data/train_sequences.csv --val-file data/val_sequences.csv --test-file data/test_internal_sequences.csv --output-dir models/tcn/results
python models/cnn_bilstm/train.py --train-file data/train_sequences.csv --val-file data/val_sequences.csv --test-file data/test_internal_sequences.csv --output-dir models/cnn_bilstm/results
python models/st_gcn/train.py --train-file data/train_sequences.csv --val-file data/val_sequences.csv --test-file data/test_internal_sequences.csv --output-dir models/st_gcn/results
```

If centroid assets are missing or if models were retrained, rebuild similarity assets:

```bash
python scripts/preprocess/build_similarity_assets.py --train-file data/train_sequences.csv --models-root models
```

## Inference (Local)
Run offline evaluation on home/generalization test data:

```bash
python scripts/evaluate/evaluate_home_set.py --test-file data/test_home_sequences.csv --models-root models --output-dir results/eval_offline_home
```

Run inference benchmarking:

```bash
python scripts/benchmark/benchmark_inference.py --input-file data/test_home_sequences.csv --models-root models --output-dir results/benchmark_inference
```

Run realtime webcam evaluation (CLI):

```bash
python scripts/realtime_eval/evaluate_realtime_webcam.py --model-name bilstm --models-root models --output-dir results/eval_realtime
```

## Streamlit App
Launch the local Streamlit interface:

```bash
streamlit run scripts/app/motionbench.py
```

In the app you can select a model, test camera capture, start a live session, and monitor predicted class, repetition count, and similarity score.

## Citations
If you publish or present this project, cite the primary libraries and frameworks used in the pipeline.

**PyTorch**
```bibtex
@misc{pytorch,
  title={PyTorch},
  howpublished={\url{https://pytorch.org/}}
}
```

**MediaPipe**
```bibtex
@misc{mediapipe,
  title={MediaPipe},
  howpublished={\url{https://mediapipe.dev/}}
}
```

**OpenCV**
```bibtex
@misc{opencv,
  title={OpenCV},
  howpublished={\url{https://opencv.org/}}
}
```

**scikit-learn**
```bibtex
@misc{scikitlearn,
  title={scikit-learn},
  howpublished={\url{https://scikit-learn.org/}}
}
```

## License
This project is released under the MIT License.
