# Thin-Film Inverse Design with CNN-Transformer

## Overview

This project investigates the inverse design of multilayer thin-film structures from their optical transmittance spectra.

The goal is to predict a multilayer structure that can reproduce a given target transmittance spectrum. The model combines a **1D CNN spectrum encoder** with an **autoregressive Transformer decoder**, followed by **Beam Search** and **L-BFGS-B continuous optimization**.

Two inverse-design problems are considered:

1. **Fixed 10-layer structure**

   - 10 alternating dielectric layers
   - Dielectric constants: ε = 3.9 and ε = 11.7
   - The model predicts the thickness of each layer
2. **Variable 2–10-layer structure**

   - The number of layers is not fixed
   - The starting material can be either ε = 3.9 or ε = 11.7
   - Materials alternate between the two dielectric constants
   - The model predicts the layer count, material sequence, and thicknesses

The optical response of each structure is calculated using the **Transfer Matrix Method (TMM)**.

---

## Method

The overall inverse-design pipeline is:

```text
Target Transmittance Spectrum
            ↓
      1D CNN Encoder
            ↓
   Spectral Feature Sequence
            ↓
 Autoregressive Transformer Decoder
            ↓
        Beam Search
            ↓
  Candidate Thin-Film Structures
            ↓
        TMM Evaluation
            ↓
   Best Discrete Structure
            ↓
       L-BFGS-B
            ↓
 Continuous Thickness Refinement
```

## Repository Structure

- `simulation.py` — Transfer Matrix Method simulation

- `generate_fixed_dataset.py` — Fixed 10-layer dataset generation

- `generate_variable_dataset.py` — Variable 2–10-layer dataset generation

- `train_fixed.ipynb` — Fixed-layer model training

- `train_variable.ipynb` — Variable-layer model training

- `test_fixed.ipynb` — Fixed-layer evaluation

- `test_variable.ipynb` — Variable-layer evaluation

## Usage

### 1. Generate Dataset

Run the corresponding dataset generation script.

### 2. Train Model

Set the dataset path in the training notebook:

DATA_DIR = "path/to/dataset"

### 3. Test Model

Set the trained model path:

MODEL_PATH = "path/to/model.pth"