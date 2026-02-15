# PyTorch-Based PPE Detection - Sprint 1 Codebase

## Project Overview
This is a PyTorch-based Personal Protective Equipment (PPE) detection system with image analysis capabilities. Features a simple CNN model and visualization tools for displaying images and statistical analysis.

## Features
- ✅ PyTorch CNN Model (SimplePPEModel with 100K+ parameters)
- ✅ Image data loading and display
- ✅ Statistical visualization and graphs
- ✅ Sample dataset generation
- ✅ PyTorch tensor support
- ✅ Dataset analysis tools

## Project Structure

```
Codebase_Sprint1/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py          # Data loading utilities
│   │   ├── augmentation.py    # Data augmentation pipeline
│   │   └── preprocessing.py   # Image preprocessing
│   ├── models/
│   │   ├── __init__.py
│   │   ├── ppemodel.py        # Transfer learning based model
│   │   └── yolov3.py          # YOLOv3 implementation
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── helpers.py         # Utility functions
│   │   └── metrics.py         # Evaluation metrics
│   └── train.py               # Trainer class
├── notebooks/
│   └── (Jupyter notebooks for exploration)
├── main.py                   # Main training script
├── inference.py              # Inference script
├── config.yaml              # Configuration file
├── generate_dataset.py      # Sample dataset generator
└── requirements.txt         # Python dependencies

Dataset_Sample/
├── images/
│   ├── image_0000.jpg
│   ├── image_0001.jpg
│   └── ...
└── annotations/
    ├── image_0000.txt (YOLO format)
    ├── image_0001.txt
    └── ...
```

## Installation

1. **Clone the repository:**
   ```bash
   cd PyTorch-Based-PPE-Detection
   ```

2. **Install dependencies:**
   ```bash
   pip install -r Codebase_Sprint1/requirements.txt
   ```

3. **Generate sample dataset:**
   ```bash
   cd Codebase_Sprint1
   python generate_dataset.py
   ```

## Usage

### Training

```bash
# Train with default configuration
python main.py --model simple --epochs 50 --batch_size 32

# Train with custom parameters
python main.py --model resnet --epochs 100 --learning_rate 0.0005

# Available models: simple, resnet, yolov3
```

### Inference

```bash
# Predict on a single image
python inference.py --model checkpoints/best_model.pth --image path/to/image.jpg

# Predict on a directory of images
python inference.py --model checkpoints/best_model.pth --image_dir path/to/images/
```

## PPE Classes

The model detects and classifies the following PPE categories:
- **Helmet**: Head protection equipment
- **Vest**: Body protection equipment (safety vest)
- **No PPE**: No protective equipment detected

## Data Format

### Images
- Format: JPEG or PNG
- Recommended size: 640 x 640 pixels
- Normalization: ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### Annotations (YOLO Format)
Each annotation file has one line per bounding box:
```
<class_id> <x_center> <y_center> <width> <height>
```
Where coordinates are normalized (0-1) relative to image size.

Example:
```
0 0.5 0.5 0.3 0.4
1 0.7 0.6 0.2 0.35
```

## Model Architectures

### 1. SimplePPEClassifier
- Lightweight CNN-based architecture
- 3 convolutional blocks + FC layers
- Fast inference, suitable for real-time applications
- ~1M parameters

### 2. PPEDetectionModel
- ResNet50 backbone with Feature Pyramid Network (FPN)
- Transfer learning with pretrained weights
- Suitable for detection tasks
- ~25M parameters

### 3. YOLOv3
- Full YOLOv3 implementation
- Multi-scale detection
- Darknet-53 backbone
- State-of-the-art for object detection

## Configuration

Edit `config.yaml` to customize:
- Dataset paths
- Model parameters
- Training hyperparameters
- Augmentation settings
- Logging preferences

## Module Documentation

### Data Loading (`src/data/loader.py`)
- `PPEDataset`: Custom dataset class supporting YOLO format
- `PPEDataLoader`: Utility for creating PyTorch DataLoaders

### Data Augmentation (`src/data/augmentation.py`)
- `DataAugmenter`: Comprehensive augmentation pipeline using Albumentations
- Supports: flipping, rotation, brightness, blur, noise, etc.

### Preprocessing (`src/data/preprocessing.py`)
- `PreprocessingPipeline`: Image loading, resizing, normalization
- Bounding box scaling and clipping utilities

### Models (`src/models/`)
- `PPEDetectionModel`: Transfer learning model with FPN
- `SimplePPEClassifier`: Lightweight CNN classifier
- `YOLOv3 & YOLOv3Lite`: Object detection architectures

### Training (`src/train.py`)
- `Trainer`: Complete training loop with validation
- Learning rate scheduling
- Checkpoint management
- Training history tracking

### Utilities (`src/utils/`)
- **helpers.py**: Device management, seed setting, model checkpointing
- **metrics.py**: Comprehensive evaluation metrics (accuracy, precision, recall, F1, IoU)

## Sprint 1 Completed Tasks

✅ Project structure setup
✅ Data loading and augmentation pipeline
✅ Image preprocessing utilities
✅ Model architecture implementation (Multiple models)
✅ Training framework with validation
✅ Inference script
✅ Evaluation metrics
✅ Configuration management
✅ Sample dataset generator
✅ Comprehensive documentation

## Next Steps (Sprint 2)

- [ ] Real dataset integration
- [ ] Model fine-tuning and optimization
- [ ] Advanced data augmentation techniques
- [ ] Model ensemble methods
- [ ] Performance optimization for deployment
- [ ] Web API for inference
- [ ] Mobile deployment considerations

## Requirements

- Python 3.8+
- PyTorch 2.0+
- TorchVision 0.15+
- OpenCV
- NumPy, Pandas
- Albumentations (for advanced augmentation)

## Team Information

**Group**: XX

## References

- [YOLOv3 Paper](https://arxiv.org/abs/1804.02767)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [PyTorch Documentation](https://pytorch.org/docs/)

## License

This project is for educational purposes.

## Support

For issues or questions, please refer to the project documentation or contact the team.
