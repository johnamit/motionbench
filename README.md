<img src="assets/motionbench.png" alt="MotionBench" width="900"><br>

[<img src="https://img.shields.io/badge/HuggingFace-Dataset-black?style=for-the-badge&logo=huggingface&logoColor=FFD21E&labelColor=ff7f1e" alt="View Dataset on Hugging Face"/>](https://huggingface.co/datasets/johnamit/motionbench-data)
&nbsp;&nbsp;
[<img src="https://img.shields.io/badge/HuggingFace-Models-black?style=for-the-badge&logo=huggingface&logoColor=FFD21E&labelColor=ff7f1e" alt="View Models on Hugging Face"/>](https://huggingface.co/johnamit/motionbench-models)
&nbsp;&nbsp;
[<img src="https://img.shields.io/badge/HuggingFace-Live App-black?style=for-the-badge&logo=huggingface&logoColor=FFD21E&labelColor=ff7f1e" alt="View Demo on Hugging Face"/>](https://huggingface.co/spaces/johnamit/motionbench)
&nbsp;&nbsp;
[<img src="https://img.shields.io/badge/Google_Drive-Demo Videos-black?style=for-the-badge&logo=google%20drive&logoColor=white&labelColor=4285F4" alt="View Videos on Drive"/>](https://drive.google.com/drive/folders/1DHvUd81QcKR6cVVvB1eRToOgxgm29lU_?usp=sharing)

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

Core work happens in `data/`, `models/`, `scripts/`, and `results/`. Older or non-essential files are moved to `archive/` to keep the main repository clear and easy to review.

## Dataset
This repo stays lightweight on GitHub. Download the dataset files from Hugging Face and place them in `data/`.

```bash
git clone https://huggingface.co/datasets/johnamit/motionbench-data data
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

```bash
git clone https://huggingface.co/johnamit/motionbench-models models
```

This project includes six sequence models with different strengths. Some are strong on temporal memory, some are better for latency, and some are better at capturing structured feature relationships.

**BiLSTM:**
The bidirectional LSTM processes each sequence in forward and backward directions within the input window, so the classifier can use context from both ends of the motion segment. This helps when important movement details are spread across the whole sequence, not just a single frame.

**LSTM:**
Unidirectional LSTM reads movement step by step in time. It is a simple and reliable sequence model, so it works well as a strong baseline for exercise classification while keeping runtime reasonable.

**GRU:**
The GRU uses gating similar to LSTM but with fewer internal components, which can reduce parameter count and improve efficiency. In practice, it is a strong candidate when you want robust sequence modeling with lighter recurrent overhead.

**TCN:**
The temporal convolutional network uses dilated 1D convolutions and residual blocks to to learn patterns over short and long time ranges. Because convolutional operations are parallelizable, it is often fast at inference, which makes it a good option when responsiveness matters.

**CNN-BiLSTM:**
This hybrid architecture first applies temporal convolutions to capture short local motion patterns, then a BiLSTM models how those patterns evolve over time. This gives both local detail and sequence context.

**ST-GCN-inspired (feature-graph variant):**
This ST-GCN-style model treats features as connected nodes and learns both their relationships and how they change over time. It can help when interactions between pose features are important for classification.

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

Please also find the [live demo](https://huggingface.co/spaces/johnamit/motionbench) of the streamlit app hosted on HuggingFace spaces via Docker. 

## Citations

**Bidirectional Long Short-Term Memory (BiLSTM)**
```bibtex
@article{riccio2024real,
  title={Real-time fitness exercise classification and counting from video frames},
  author={Riccio, Riccardo},
  journal={arXiv preprint arXiv:2411.11548},
  year={2024}
}
```

**Gated Recurrent Unit (GRU)**
```bibtex
@article{chung2014empirical,
  title={Empirical evaluation of gated recurrent neural networks on sequence modeling},
  author={Chung, Junyoung and Gulcehre, Caglar and Cho, KyungHyun and Bengio, Yoshua},
  journal={arXiv preprint arXiv:1412.3555},
  year={2014}
}
```

**Temporal Convolutional Network (TCN)**
```bibtex
@inproceedings{lea2017temporal,
  title={Temporal convolutional networks for action segmentation and detection},
  author={Lea, Colin and Flynn, Michael D and Vidal, Rene and Reiter, Austin and Hager, Gregory D},
  booktitle={proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={156--165},
  year={2017}
}
```

**Spatial Temporal Graph Convolutional Network**
```bibtex
@inproceedings{yan2018spatial,
  title={Spatial temporal graph convolutional networks for skeleton-based action recognition},
  author={Yan, Sijie and Xiong, Yuanjun and Lin, Dahua},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  volume={32},
  number={1},
  year={2018}
}
```

**CNN BiLSTM Hybrid**
```bibtex
@online{dhomane2024cnnbilstm,
  author  = {Shreyas Dhomane},
  title   = {CNN + BiLSTM Architecture: A Practical Guide},
  year    = {2024},
  month   = oct,
  day     = {23},
  url     = {https://medium.com/@shreyas.dhomane22/cnn-bilstm-architecture-a-practical-guide-c81829022820},
  note    = {Medium article. Accessed: 2026-04-22}
}
```

## License
This project is released under the MIT License.