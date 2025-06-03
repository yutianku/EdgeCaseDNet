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


## License

This project is licensed under [LICENSE].

## Contact

For questions, please contact us via Issues 


##--------------------------------------------------------------------------

## 📄 Documentation -YOLO

See below for quickstart installation and usage examples. For comprehensive guidance on training, validation, prediction, and deployment, refer to our full [Ultralytics Docs](https://docs.ultralytics.com/).

<details open>
<summary>Install</summary>

Install the `ultralytics` package, including all [requirements](https://github.com/ultralytics/ultralytics/blob/main/pyproject.toml), in a [**Python>=3.8**](https://www.python.org/) environment with [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/).

[![PyPI - Version](https://img.shields.io/pypi/v/ultralytics?logo=pypi&logoColor=white)](https://pypi.org/project/ultralytics/) [![Ultralytics Downloads](https://static.pepy.tech/badge/ultralytics)](https://www.pepy.tech/projects/ultralytics) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ultralytics?logo=python&logoColor=gold)](https://pypi.org/project/ultralytics/)

```bash
pip install ultralytics
```

For alternative installation methods, including [Conda](https://anaconda.org/conda-forge/ultralytics), [Docker](https://hub.docker.com/r/ultralytics/ultralytics), and building from source via Git, please consult the [Quickstart Guide](https://docs.ultralytics.com/quickstart/).

[![Conda Version](https://img.shields.io/conda/vn/conda-forge/ultralytics?logo=condaforge)](https://anaconda.org/conda-forge/ultralytics) [![Docker Image Version](https://img.shields.io/docker/v/ultralytics/ultralytics?sort=semver&logo=docker)](https://hub.docker.com/r/ultralytics/ultralytics) [![Ultralytics Docker Pulls](https://img.shields.io/docker/pulls/ultralytics/ultralytics?logo=docker)](https://hub.docker.com/r/ultralytics/ultralytics)

</details>

<details open>
<summary>Usage</summary>

### CLI

You can use Ultralytics YOLO directly from the Command Line Interface (CLI) with the `yolo` command:

```bash
# Predict using a pretrained YOLO model (e.g., YOLO11n) on an image
yolo predict model=yolo11n.pt source='https://ultralytics.com/images/bus.jpg'
```

The `yolo` command supports various tasks and modes, accepting additional arguments like `imgsz=640`. Explore the YOLO [CLI Docs](https://docs.ultralytics.com/usage/cli/) for more examples.

### Python

Ultralytics YOLO can also be integrated directly into your Python projects. It accepts the same [configuration arguments](https://docs.ultralytics.com/usage/cfg/) as the CLI:

```python
from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Train the model on the COCO8 dataset for 100 epochs
train_results = model.train(
    data="coco8.yaml",  # Path to dataset configuration file
    epochs=100,  # Number of training epochs
    imgsz=640,  # Image size for training
    device="cpu",  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
)

# Evaluate the model's performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("path/to/image.jpg")  # Predict on an image
results[0].show()  # Display results

# Export the model to ONNX format for deployment
path = model.export(format="onnx")  # Returns the path to the exported model
```

Discover more examples in the YOLO [Python Docs](https://docs.ultralytics.com/usage/python/).


