# Generative Adversarial Learning of Sinkhorn Algorithm Initializations

This repository implements the method described in the paper 'Generative Adversarial Learning of Sinkhorn Algorithm Initializations'.

## Overview

This project aims at warm-starting the Sinkhorn algorithm with initializations computed by a neural network. The network is trained in an adversarial fashion similar to a GAN using a second, generating neural network. This approach accelerates the convergence of the Sinkhorn algorithm for computing optimal transport distances.

The work is based on two Master's theses:
- 'A Sinkhorn-NN Hybrid Algorithm' by Jonathan Geuter
- 'Learning Optimal Transport Solutions with Deep Neural Networks' by Ingimar Tomasson

## Installation

### Requirements

This package requires Python 3.x and is CUDA compatible for GPU acceleration.

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd learn_sink
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
- numpy==1.23.5
- matplotlib==3.7.1
- tqdm==4.66.1
- torch==2.1.0
- jax==0.4.16
- ott-jax==0.4.4
- torchvision==0.16.0
- skimage==0.19.3

## Data Preparation

The experiments use several image datasets: MNIST, CIFAR-10, LFW (Labeled Faces in the Wild), and QuickDraw.

### Option 1: Download Pre-processed Data

Download the pre-processed test data from [this Google Drive folder](https://drive.google.com/drive/folders/1o5pz9-Zhr1-7s1QvPnOiTCVGWULmfC9f?usp=drive_link) and place the `.pt` files in a `./data` directory.

### Option 2: Generate Data Locally

Run the data generation script to download and process the datasets:

```bash
python make_data.py
```

This will:
- Download MNIST, CIFAR-10, LFW, and QuickDraw datasets
- Process and save them as PyTorch tensors in the `./data` directory
- Generate 10,000 samples for each dataset

## Usage

### Running Experiments

To reproduce the results from the paper, simply execute:

```bash
python experiment.py
```

This will:
1. Load and preprocess the test datasets
2. Initialize the generator (GenNet) and predictor (PredNet) neural networks
3. Train both networks in an adversarial manner
4. Save results, checkpoints, and plots in a timestamped `./results/experiment_<timestamp>` folder

### Project Structure

```
learn_sink/
├── experiment.py          # Main experiment script
├── make_data.py          # Data download and preprocessing
├── requirements.txt      # Package dependencies
├── src/
│   ├── checkpoint.py     # Model checkpointing utilities
│   ├── data_funcs.py     # Data loading and preprocessing functions
│   ├── geometry.py       # Cost matrix computation
│   ├── loss.py          # Loss functions (Hilbert projection, MSE)
│   ├── nets.py          # Neural network architectures (GenNet, PredNet)
│   ├── plot.py          # Visualization utilities
│   ├── sinkhorn.py      # Sinkhorn algorithm implementation
│   ├── test_funcs.py    # Testing and evaluation functions
│   └── train.py         # Training loop implementation
└── tests/
    └── test_sinkhorn.py  # Unit tests for Sinkhorn algorithm
```

## Hardware Requirements

- The code automatically detects and uses CUDA-enabled GPUs if available
- Falls back to CPU if no GPU is detected
- Recommended: NVIDIA GPU with at least 8GB VRAM for faster training

## Output

Results are saved in timestamped folders under `./results/` and include:
- Model checkpoints (GenNet and PredNet weights)
- Training metrics and loss curves
- Test set performance evaluations
- Experiment configuration details

## Citation

If you use this code in your research, please cite the paper 'Generative Adversarial Learning of Sinkhorn Algorithm Initializations'. 
