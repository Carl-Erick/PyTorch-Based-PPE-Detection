Sample dataset structure and format information

This Dataset_Sample folder contains:
- images/: Folder containing sample PPE detection images (JPEG/PNG format)
- annotations/: Folder containing YOLO format annotations (one .txt file per image)

YOLO Annotation Format:
Each line in a .txt file represents one bounding box:
<class_id> <x_center> <y_center> <width> <height>

Where:
- class_id: 0=helmet, 1=vest, 2=no_ppe
- x_center, y_center: Center coordinates (0-1 normalized)
- width, height: Box dimensions (0-1 normalized)

Example annotation for image with helmet:
0 0.5 0.5 0.3 0.4

Example annotation for image with vest:
1 0.7 0.6 0.2 0.35

To generate sample images with annotations, run:
python generate_dataset.py

This will create 20 synthetic images with random bounding boxes for training/testing purposes.

For production use:
- Replace with real PPE dataset
- Ensure proper annotation format
- Validate dataset quality before training
- Split data into train/val/test sets

Recommended image sizes: 640x640 (can be adjusted in config.yaml)
Minimum images for training: 100+ per class
Recommended total: 1000+ images
