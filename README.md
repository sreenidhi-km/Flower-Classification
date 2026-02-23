# Flower Classification Using PyTorch

A deep learning project to classify **102 different types of flowers** using Transfer Learning with ResNet50 in PyTorch.

## Overview

This project implements a flower image classifier that can identify 102 different flower categories using the [102 Category Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) from the Visual Geometry Group at Oxford.

### Key Features

- **Transfer Learning** with pre-trained ResNet50 backbone
- **Data Augmentation** (random rotation, cropping, flipping, color jitter)
- **Two-Phase Training**: Frozen backbone → Full fine-tuning
- **Modern Training Techniques**:
  - OneCycleLR scheduler for optimal learning rate scheduling
  - Automatic Mixed Precision (AMP) for faster GPU training
  - Gradient clipping for stable training
  - Label smoothing for better generalization
- **Comprehensive Evaluation**: Accuracy, confusion matrix, per-class metrics

## Model Performance

- **Validation Accuracy**: ~96%
- **Test Accuracy**: ~95%

## Project Structure

```
Flower-Classification/
├── flower_classification_PyTorch.ipynb  # Main notebook with full pipeline
├── README.md                            # This file
├── requirements.txt                     # Python dependencies
├── .gitignore                          # Git ignore rules
└── cat_to_name.json                    # (Optional) Flower category to name mapping
```

## Dataset

The project uses the **102 Category Flower Dataset**:
- **Source**: [VGG Oxford](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)
- **Classes**: 102 flower categories
- **Structure**: Images organized in folders by class (1-102)

### Dataset Setup

1. Download the dataset from the source above
2. Organize into the following structure:
   ```
   flower_data/
   ├── train/
   │   ├── 1/
   │   ├── 2/
   │   └── ... (102 class folders)
   ├── valid/
   │   ├── 1/
   │   ├── 2/
   │   └── ... (102 class folders)
   └── test/
       └── 0/  (all test images in one folder for ImageFolder compatibility)
   ```

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/sreenidhi-km/Flower-Classification.git
   cd Flower-Classification
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:
   - **Google Colab** (recommended for GPU): Upload the notebook and follow the setup cells
   - **Local**: Launch Jupyter and open `flower_classification_PyTorch.ipynb`

## Usage

### Training

The notebook provides a complete pipeline:
1. Mount Google Drive / Set up local data paths
2. Load and preprocess the dataset
3. Train the model (Phase 1: frozen backbone, Phase 2: fine-tuning)
4. Evaluate on validation set
5. Save the trained model

### Inference

```python
# Load the trained model
checkpoint = torch.load('flower-classifier-final.pth', map_location=device)
model = FlowersModel(checkpoint['num_classes']).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predict on a new image
prediction = predict_image(img_tensor, model, dataset)
```

## Model Architecture

- **Backbone**: ResNet50 (pre-trained on ImageNet)
- **Classifier Head**:
  - Dropout (0.3)
  - Linear (2048 → 512)
  - ReLU + BatchNorm
  - Dropout (0.15)
  - Linear (512 → 102)

## Training Configuration

| Parameter | Phase 1 (Frozen) | Phase 2 (Fine-tune) |
|-----------|------------------|---------------------|
| Epochs | 15 | 5 |
| Max LR | 3e-3 | 1e-4 |
| Weight Decay | 1e-2 | 1e-2 |
| Batch Size | 128 | 128 |
| Optimizer | AdamW | AdamW |

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)

See `requirements.txt` for full list of dependencies.

## Future Improvements

- [ ] More aggressive data augmentation
- [ ] Experiment with other backbones (EfficientNet, ViT)
- [ ] Hyperparameter tuning with Optuna
- [ ] Weights & Biases integration for experiment tracking
- [ ] Model quantization for deployment

## License

This project is open source and available under the MIT License.

## Acknowledgments

- [102 Category Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) by VGG Oxford
- PyTorch and torchvision teams
- [Udacity](https://www.udacity.com/) for the dataset preprocessing approach
