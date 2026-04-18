# Crop Disease Detection — CNN with ResNet

A deep learning model that detects diseases in potato and tomato crops from leaf images, built using a ResNet-based CNN architecture.

## What It Does

Classifies crop leaf images into categories:
- **Potato** — Early Blight, Late Blight, Healthy
- **Tomato** — Multiple disease classes, Healthy

## Tech Stack

- **Model:** ResNet (CNN-based transfer learning)
- **Framework:** TensorFlow / Keras
- **MLOps:** Model versioning and experiment tracking
- **Dataset:** PlantVillage dataset

## Model Architecture

Uses a pre-trained ResNet backbone with custom classification head fine-tuned on crop disease images. The transfer learning approach achieves high accuracy with relatively small training data.

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python train.py

# Run prediction on an image
python predict.py --image path/to/leaf.jpg
```

## Results

The model distinguishes healthy crops from diseased ones, enabling early detection to reduce crop losses.
