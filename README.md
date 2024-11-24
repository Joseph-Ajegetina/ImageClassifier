Here's the complete README properly formatted in markdown:

# Deep Learning Image Classifier for Flower Species

## Project Overview

This project implements a command line application for training and predicting flower species using deep learning. The model trains on a dataset of images and uses transfer learning from either ResNet18 or VGG13 architectures to achieve high accuracy in flower classification.

## Getting Started

### Prerequisites

* Python 3
* PyTorch
* torchvision 
* PIL
* numpy
* matplotlib

### Installation

1. Clone this repository
2. Install required packages:
   ```bash
   pip install torch torchvision PIL numpy matplotlib
   ```

## Usage

### Training the Model

Train a new network on a data set with train.py:

```bash
python train.py data_directory --save_dir save_directory --arch "vgg13" --learning_rate 0.01  --epochs 20 --gpu
```

Options:
* `data_directory`: Path to training data (required)
* `--save_dir`: Directory to save checkpoints (default: current directory)
* `--arch`: Model architecture - "vgg13" or "resnet18" (default: "resnet18") 
* `--learning_rate`: Set learning rate (default: 0.003)
* `--hidden_units`: Hidden units for classifier (default: [1024, 512, 256])
* `--epochs`: Number of training epochs (default: 3)
* `--gpu`: Use GPU for training if available (default: False)

### Making Predictions

Predict flower names from images using predict.py:

```bash
python predict.py /path/to/image checkpoint --top_k 3 --category_names cat_to_name.json --gpu
```

Options:
* Image path (required)
* Checkpoint path (required)
* `--top_k`: Return top K predictions (default: 1)
* `--category_names`: Use category names JSON file (default: cat_to_name.json)
* `--gpu`: Use GPU for inference if available (default: False)

## Project Structure

```
├── train.py           # Script for training the network
├── predict.py         # Script for making predictions
├── cat_to_name.json   # Mapping of categories to flower names  
└── README.md
```

## Model Architecture 

The project offers two pre-trained model architectures:

1. ResNet18 (default)
   - Pretrained on ImageNet
   - Custom classifier added with configurable hidden units
   - Dropout added for regularization

2. VGG13
   - Pretrained on ImageNet 
   - Modified classifier with configurable hidden units
   - Dropout layers to prevent overfitting

## Data Processing

- Images are loaded using torchvision's ImageFolder
- Training transformations include:
  - Random rotation
  - Random resizing & cropping
  - Random horizontal flips
  - Normalization
- Validation/Testing transformations:
  - Resizing 
  - Center crop
  - Normalization

## Training Process 

1. Loads pretrained model and freezes feature parameters
2. Adds new classifier for flower categories
3. Trains using:
   - Adam optimizer
   - NLLLoss criterion
   - Learning rate scheduler
4. Validates accuracy during training
5. Saves checkpoint with model & optimizer state

## Checkpointing

Saved checkpoints include:
- Model state dict
- Optimizer state dict
- Class to index mapping
- Epoch completed
- Architecture used
- Hidden layer units
- Learning rate


## Acknowledgments

- Project completed as part of the Udacity AI Programming with Python Nanodegree
- Architecture implementations based on torchvision models
