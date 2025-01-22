# Mango Visual Search Training & Evaluation Pipeline

A modular pipeline for training, vectorizing, visualizing, and evaluating image classification models using PyTorch, specifically designed for visual search applications.

## Overview

This pipeline provides an end-to-end solution for:
* Training a ResNet34 model on custom image datasets
* Generating vector embeddings for ground truth images
* Visualizing embeddings in 3D space
* Evaluating model performance across different image types

## Installation

### Prerequisites

* Python 3.8+
* CUDA-capable GPU (recommended)

### Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/Riley-Livingston/mango-visual-search-training-eval-pipeline.git
    cd mango-visual-search-training-eval-pipeline
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Quick Start

1. Create a configuration file (see `config_example.yaml`)
2. Run the pipeline

### Directory Structure

```plaintext
mango-visual-search-training-eval-pipeline/
├── data/
│   ├── training_images/
│   ├── ground_truth_images/
│   └── test_sets/
├── models/
├── visualizations/
├── src/
│   ├── model_training.py
│   ├── vectorize_ground_truth.py
│   ├── vector_viz.py
│   ├── model_eval.py
│   └── run_pipeline.py
├── config_example.yaml
├── requirements.txt
└── README.md
```

## Pipeline Components

### 1. Model Training (`model_training.py`)
* Implements ResNet34-based image classifier
* Supports early stopping and learning rate scheduling
* Configurable batch size and workers

### 2. Vectorization (`vectorize_ground_truth.py`)
* Generates vector embeddings for ground truth images
* Supports batch processing
* Includes checkpoint saving and resumption

### 3. Visualization (`vector_viz.py`)
* Creates 3D visualizations of vector embeddings
* PCA-based dimensionality reduction
* Interactive HTML output

### 4. Evaluation (`model_eval.py`)
* Comprehensive model evaluation
* Support for multiple test sets
* Detailed accuracy metrics

## Requirements

```txt
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.19.2
pandas>=1.2.4
scikit-learn>=0.24.2
plotly>=4.14.3
pillow>=8.2.0
pyyaml>=5.4.1
tqdm>=4.61.0
matplotlib>=3.4.2
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

* ResNet architecture by Microsoft Research
* PyTorch team for the deep learning framework
* scikit-learn team for PCA implementation

## Contact

Riley Livingston - [GitHub](https://github.com/Riley-Livingston)

Project Link: [https://github.com/Riley-Livingston/mango-visual-search-training-eval-pipeline](https://github.com/Riley-Livingston/mango-visual-search-training-eval-pipeline)