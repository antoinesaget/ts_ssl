# Time Series Self-Supervised Learning (TS-SSL)

A framework for self-supervised learning on time series data, supporting various contrastive learning approaches and data augmentation techniques.

## Citation

🚧 Work in Progress 🚧

## Overview

RS-TS-SSL is a PyTorch-based framework designed for self-supervised learning on time series data, with a focus on remote sensing applications. The framework supports:

- Multiple contrastive learning frameworks (SimCLR, MoCo, BYOL, VICReg)
- Various time series augmentation techniques (Jitter, Masking, Cropping, Resampling)
- Flexible encoder architectures
- Comprehensive evaluation and fine-tuning capabilities

## Project Structure

```
ts_ssl/
├── config/           # Configuration files (Hydra)
│   ├── location/    # Dataset path configurations
│   └── ...          # Other config categories
├── runs/            # Training and testing scripts
└── ts_ssl/       # Main package
```

## Installation

### 1 - Environment Setup

Choose the appropriate environment based on your hardware:

1. **CPU-only** (Basic setup):
```bash
conda env create -f environment-minimal-cpu.yml
conda activate ts_ssl_cpu
```

2. **GPU** (Faster training):
```bash
conda env create -f environment-minimal-gpu.yml
conda activate ts_ssl_gpu
```

3. **GPU + RAPIDS cuML** (Fastest, with accelerated evaluation using NVIDIA rapids cuml's logistic regression implementation instead of scikit-learn logistic regression):
```bash
conda env create -f environment-minimal-gpu-cuml.yml
conda activate ts_ssl
```

### 2 - Data Setup

Install Hugging Face datasets:
```bash
pip install datasets
```

Download the sample dataset:
```python
python preload_datasets.py francecrops_nano
```

(Optional) If you want to download to a different location, change cache_dir in `ts_ssl/config/dataset/francecrops_nano.yaml`

Note: only francecrops_nano is currently supported.
Warning: currently the train, validation and test data are the same in francecrops_nano.

More information on the dataset can be found here: https://huggingface.co/datasets/saget-antoine/francecrops

### 3 - Testing the Installation

CPU test:
```bash
python ts_ssl/train.py \
    dataset=francecrops_nano \
    training.max_examples_seen=10000 \
    training.val_check_interval=100 \
    post_training_eval=true \
    model.encoder.n_filters=64 \
    model.encoder.embedding_dim=128 \
    training.batch_size=64 \
    num_workers=8 \
    "validation.samples_per_class=[5,10]" \
    "eval.samples_per_class=[5,10]" \
    device=cpu
```

GPU test:
```bash
python ts_ssl/train.py \
    dataset=francecrops_nano \
    training.max_examples_seen=40000 \
    training.val_check_interval=200 \
    post_training_eval=true \
    model.encoder.n_filters=64 \
    model.encoder.embedding_dim=128 \
    training.batch_size=64 \
    num_workers=8 \
    "validation.samples_per_class=[5,10]" \
    "eval.samples_per_class=[5,10]" \
    device=cuda
```

Low-end GPU (4GB RAM) test:
```bash
python ts_ssl/train.py \
    dataset=francecrops_nano \
    training.max_examples_seen=40000 \
    training.val_check_interval=200 \
    post_training_eval=true \
    model.encoder.n_filters=48 \
    model.encoder.embedding_dim=96 \
    training.batch_size=96 \
    num_workers=8 \
    "validation.samples_per_class=[5,10]" \
    "eval.samples_per_class=[5,10]" \
    device=cuda
```

Note: When using the GPU+cuML environment, the test will automatically use cuML's LogisticRegression for evaluation (check logs for "Using cuML LogisticRegression").

## Training

🚧 Work in Progress 🚧

Launch a training run:
```bash
python ts_ssl/train.py \
    model=simclr \
    dataset=francecrops \
    augmentations=resampling
```

## Evaluation

🚧 Work in Progress 🚧

Evaluate a trained model:
```bash
python ts_ssl/evaluate.py \
    checkpoint_path=outputs/YOUR_RUN_DIR/checkpoints/last.ckpt
```

## Fine-tuning

🚧 Work in Progress 🚧

Fine-tune a pre-trained model:
```bash
python ts_ssl/finetune.py \
    checkpoint_path=outputs/YOUR_RUN_DIR/checkpoints/last.ckpt \
    dataset=francecrops
```

## Optional Dependencies

Tensoboard and/or Neptune can optionally be used to track experiments.

### Tensorboard
```bash
pip install tensorboard
# add "tensorboard" to logging:enabled_loggers: list in config/config.yaml 
```

### Neptune.ai
```bash
pip install neptune

# Configure Neptune
export NEPTUNE_API_TOKEN="your-api-token"
# Update config/config.yaml with:
# neptune:
#   enabled: true
#   project: "your-project"
```

## Configuration

🚧 Work in Progress 🚧

The project uses Hydra for configuration management. Key configuration areas:

- **Model**: Change encoder architecture and SSL framework in `config/model/`
- **Dataset**: Configure dataset parameters in `config/dataset/`
- **Augmentations**: Modify augmentation strategies in `config/augmentations/`
- **Training**: Adjust training parameters in `config/config.yaml`

For a complete list of configuration options, run:
```bash
python ts_ssl/train.py --help
```

## License

🚧 Work in Progress 🚧
