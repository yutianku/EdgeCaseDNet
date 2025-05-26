# EdgeCaseDNet
# EdgeCaseDNet: Enhanced YOLOv8 for Traffic Object Detection

EdgeCaseDNet is an improved model based on YOLOv8, specifically designed for traffic scene object detection. The project utilizes the HaarHGNet architecture to provide efficient and accurate traffic object detection through optimized network structure.

## Key Features

- Improved architecture based on YOLOv8
- Uses HaarHGNet as backbone network
- Performance optimized for traffic scenarios
- Supports detection of 6 traffic-related object classes
- High inference speed

## Detection Classes

1. Pedestrian
2. Cyclist 
3. Car
4. Truck
5. Tram
6. Tricycle

## Project Structure

```
EdgeCaseDNet/
├── detect.py          # Main detection program
├── train.py          # Training program
├── val.py            # Validation program
├── data.yaml         # Dataset configuration
├── yolov8-HaarHGNet-l.yaml  # Model configuration
└── ultralytics/      # Modified YOLOv8 core library
```

## Quick Start

### Requirements

- Python 3.8+
- PyTorch 1.7+
- CUDA 10.2+ (recommended)

### Installation

```bash
git clone https://github.com/your-username/EdgeCaseDNet.git
cd EdgeCaseDNet
pip install -r requirements.txt
```

### Usage

1. Detection example:

```python
# Using detect.py for detection
python detect.py
```

2. Training your own model:

```bash
python train.py --data data.yaml --cfg yolov8-HaarHGNet-l.yaml --weights weights/yolov8x.pt
```

3. Model validation:

```bash
python val.py --weights weights/best.pt --data data.yaml
```

## Model Configuration

Main configuration file: `yolov8-HaarHGNet-l.yaml`

```yaml
# Model architecture configuration
backbone:
  - [-1, 1, HGStem, [128]]  # HaarHGNet stem
  - [-1, 1, HGBlock, [256]] # HaarHGNet block
  ...

head:
  - [-1, 1, SPPF, [1024, 5]]  # SPP-Fast
  - [-1, 1, C2f, [512]]      # CSP Fusion
  ...
```

## Dataset

Using SODA dataset:
- Training set path: E:\SC\Yolov8\Dataset\soda\images\train
- Validation set path: E:\SC\Yolov8\Dataset\soda\images\val
- Test set path: E:\SC\Yolov8\Dataset\soda\images\test

## License

This project is licensed under [LICENSE].

## Contact

For questions, please contact us via Issues 
