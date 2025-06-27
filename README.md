# 🌸 Flower Image Classifier – Deep Learning with PyTorch

This project implements a deep learning image classifier that can identify 102 different species of flowers. It uses transfer learning on a pre-trained VGG16 architecture and is built entirely in PyTorch. The classifier can be trained, saved, and used for inference directly from the command line.

> 🎯 **Goal:** Imagine a mobile app that tells you what flower you're looking at. This project is the command-line foundation for that!

*Example: Classifying a Passion Flower using the trained model.*

---

## 📋 Table of Contents

- [Key Features](#-key-features)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Dataset](#dataset)
- [How to Use](#-how-to-use)
  - [Training the Model](#-training-the-model)
  - [Classifying an Image](#-classifying-an-image)
- [Model Architecture & Project Structure](#-model-architecture--project-structure)
- [Performance](#-performance)
- [Learnings & Future Work](#-learnings--future-work)
- [Acknowledgements](#-acknowledgements)
- [Author](#-author)

---

## ✨ Key Features

-   **Train Custom Classifier**: Use `train.py` to train a new classifier on the flower dataset and save model checkpoints.
-   **Predict with New Images**: Use `predict.py` to load a saved checkpoint and classify any flower image.
-   **Transfer Learning**: Leverages the powerful, pre-trained VGG16 model to achieve high accuracy with less data.
-   **Command-Line Interface**: All functionality is accessible via a flexible and easy-to-use CLI.
-   **GPU Acceleration**: Automatically uses the GPU for training and inference if a CUDA-enabled device is available.
-   **Modular Codebase**: A clean separation of concerns between model definition, training logic, and prediction.

---

## 🚀 Getting Started

Follow these steps to set up the project environment on your local machine.

### Prerequisites

-   Python 3.9+
-   `pip` for package management
-   A virtual environment is highly recommended.

### Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/devanshii-03/image-classification.git
    cd image-classification
    ```

2.  **Install Dependencies**
    ```bash
    pip install torch torchvision numpy matplotlib pillow argparse
    ```

### Dataset

The model is trained on the **Oxford 102 Flower Dataset**.

1.  **Download the Dataset** from the [official source](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html).
2.  **Extract the Dataset** into a `flowers/` folder within your project directory.

---

## 🕹️ How to Use

All scripts are located in the `Part 2 files/` directory.

### ➤ Training the Model

Use `train.py` to train the classifier. You can customize the architecture, learning rate, epochs, and other hyperparameters.

**Example Command:**
```bash
python "Part 2 files/train.py" flowers --arch vgg16 --learning_rate 0.001 --epochs 5 --gpu
```

**Available Arguments:**

  - `data_directory`: (Positional) Path to the flower image dataset (e.g., `flowers/`).
  - `--save_dir`: Directory to save model checkpoints (default: current directory).
  - `--arch`: Model architecture (default: `vgg16`).
  - `--learning_rate`: Learning rate for the optimizer (default: `0.001`).
  - `--hidden_units`: Number of units in the hidden layer (default: `512`).
  - `--epochs`: Number of training epochs (default: `5`).
  - `--gpu`: Use GPU for training (flag, no value needed).

A checkpoint file (e.g., `checkpoint.pth`) will be saved upon completion.

### ➤ Classifying an Image

Use the `predict.py` script to predict the class (or top K classes) of a flower image.

**Basic Prediction Command:**

```bash
python "Part 2 files/predict.py" path/to/your/flower.jpg checkpoint.pth --top_k 5 --category_names cat_to_name.json --gpu
```

**Available Arguments:**

  - `path_to_image`: (Positional) Path to the image you want to classify.
  - `checkpoint`: (Positional) Path to the saved model checkpoint.
  - `--top_k`: Return top K most likely classes (default: `1`).
  - `--category_names`: Path to the JSON file mapping categories to real names (e.g., `cat_to_name.json`).
  - `--gpu`: Use GPU for inference (flag, no value needed).

-----

## 🧠 Model Architecture & Project Structure

<details>
<summary>
<strong>Model Overview</strong> (Click to expand)
</summary>

  - **Base Model**: VGG16 (pre-trained on ImageNet).
  - **Transfer Learning**: Convolutional layers are frozen to retain learned features. The classifier part is replaced and trained from scratch.
  - **Custom Classifier**:
      - Fully Connected Layer (Input features from VGG16 -\> Hidden Units)
      - ReLU Activation
      - Dropout (for regularization)
      - Fully Connected Layer (Hidden Units -\> 102 Output Classes)
      - LogSoftmax Output Layer
  - **Loss Function**: Negative Log Likelihood Loss (`nn.NLLLoss`), as it works well with `LogSoftmax`.
  - **Optimizer**: Adam optimizer.
  - **Input Image Size**: Images are resized and cropped to `224x224` pixels.

</details>

<details>
<summary>
  <strong>Project File Structure</strong> (Click to expand)</summary>

```
image-classification/
├── Completed part 1 notebook/
│   └── Flower Classifier.html   # Interactive notebook (exported to HTML)
│
├── Part 2 files/
│   ├── model.py                 # Defines model architecture, save/load checkpoints
│   ├── train.py                 # Command-line script for training the model
│   ├── util.py                  # Helper functions and data transforms
│   └── predict.py               # Command-line script for predicting with a saved model
│
├── cat_to_name.json             # Mapping from category index to flower name
├── README.md                    # This file
└── .gitignore                   # Files/directories to ignore in Git
```

</details>

-----

## 📊 Performance

The model generalizes well to unseen flower images, thanks to transfer learning and data augmentation techniques like random rotation and flipping.

| Metric              | Value |
| :------------------ | :---- |
| Training Accuracy   | \~98%  |
| Validation Accuracy | \~87%  |
| Test Accuracy       | \~85%  |

-----

## 💡 Learnings & Future Work

  - ✅ Gained practical experience with **transfer learning** in PyTorch.
  - ✅ Learned to **modularize code** into reusable components and build flexible **command-line tools**.
  - ✅ Deepened understanding of model saving/loading, data augmentation, and hyperparameter tuning.

#### Future Extensions:

  - 📱 **Deploy as an App**: Convert this into a web-based flower recognition app using **Flask** or **Streamlit**.
  - 🔬 **Experiment with Architectures**: Test other pre-trained models like ResNet or DenseNet.
  - ⚙️ **Hyperparameter Tuning**: Implement a systematic search for optimal hyperparameters using tools like Optuna.

-----

## 🙌 Acknowledgements

  - **Dataset**: [Oxford 102 Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
  - **Guidance**: Udacity – For the AI Programming with Python Nanodegree that inspired this project
  - **Framework**: PyTorch Documentation

-----

## 👩‍💻 Author

**Devanshi Nikam**

*Final Year B.Tech | AI/ML Enthusiast*

Passionate about solving real-world problems through intelligent systems.

[](https://www.google.com/search?q=https://github.com/your-username)
[](https://www.linkedin.com/in/your-profile/)

⭐ If this project helped or inspired you, please consider giving it a ⭐ on GitHub and sharing your feedback\!

```
```
