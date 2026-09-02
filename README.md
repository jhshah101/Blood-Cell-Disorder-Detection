# Hybrid CNN-Transformer with Efficient Channel Attention for White Blood Cell Classification

## Overview

This repository provides the PyTorch implementation of a **Hybrid CNN-Transformer framework integrated with Efficient Channel Attention (ECA)** for **White Blood Cell (WBC) classification** under class-imbalanced conditions.

The proposed framework combines the local feature extraction capability of **ResNet18**, the global contextual representation of **Transformer Encoder**, and **Efficient Channel Attention (ECA)** to improve minority-class recognition without relying on image augmentation. Class imbalance is handled using **class-weighted Cross-Entropy Loss** computed from the training data.

The framework also supports **cross-dataset evaluation** to assess model robustness and generalization.

---

## Features

- Hybrid CNN + Transformer architecture
- Efficient Channel Attention (ECA)
- No data augmentation
- Class-weighted Cross-Entropy Loss
- Cross-dataset evaluation
- Macro-F1 score evaluation
- Reproducible training pipeline
- PyTorch implementation

---

## Model Architecture

```
Input Image
      │
      ▼
 ResNet18 Backbone
      │
      ▼
 Efficient Channel Attention (ECA)
      │
      ▼
 Patch Token Generation
      │
      ▼
 Transformer Encoder
      │
      ▼
 Classification Head
      │
      ▼
 White Blood Cell Prediction
```

---

## Repository Structure

```
.
├── WBC_Without.py          # Main training script
├── README.md               # Documentation
├── checkpoints/            # Saved models
├── datasets/               # Dataset folders
├── results/                # Experimental results
└── requirements.txt
```

---

## Dataset Structure

The code uses the PyTorch `ImageFolder` format.

```
Dataset/

├── train/
│   ├── Basophil/
│   ├── Eosinophil/
│   ├── Lymphocyte/
│   ├── Monocyte/
│   └── Neutrophil/
│
└── test/
    ├── Basophil/
    ├── Eosinophil/
    ├── Lymphocyte/
    ├── Monocyte/
    └── Neutrophil/
```

For cross-dataset evaluation, use one dataset for training and another independent dataset for testing.

---

## Requirements

- Python 3.10+
- PyTorch
- Torchvision
- NumPy

Install dependencies:

```bash
pip install torch torchvision numpy
```

---

## Training

Edit the dataset paths in `WBC_Without.py`:

```python
TRAIN_DIR = "/path/to/train"
TEST_DIR  = "/path/to/test"
```

Run:

```bash
python WBC_Without.py
```

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Input Size | 224 × 224 |
| Batch Size | 32 |
| Epochs | 20 |
| Optimizer | AdamW |
| Learning Rate | 3e-4 |
| Backbone | ResNet18 |
| Attention Module | Efficient Channel Attention (ECA) |
| Loss Function | Class-Weighted Cross-Entropy |
| Evaluation Metric | Accuracy & Macro-F1 |

---

## Evaluation

The implementation reports:

- Classification Accuracy
- Macro-F1 Score
- Confusion Matrix

The best model is selected according to **Macro-F1 Score**.

---

## Output

The trained model is automatically saved as

```
wbc_eca_cnn_transformer.pth
```

---

## Reproducibility

Random seeds are fixed to improve reproducibility.

```python
set_seed(42)
```

---

## Citation

If you use this repository in your research, please cite the corresponding manuscript.

```bibtex
@article{Jamal2026,
  title={A Hybrid ViT-ECA Framework with Adaptive Loss Reweighting for Minority White Blood Cell Classification},
  author={Fouzia Jabeen, Jamal Hussain Shah and co-authors},
  journal={PLOS ONE},
  year={2026},
  note={Under Review}
}
```

---

## License

This project is intended for academic and research purposes.

---


COMSATS University Islamabad, Wah Campus

For questions regarding the implementation, please open a GitHub Issue or contact the corresponding author.
