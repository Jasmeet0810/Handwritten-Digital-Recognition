# Handwritten Digit Classifier with PyTorch

This project implements a simple yet effective feed-forward neural network using **PyTorch** to classify handwritten digits from the **MNIST dataset**. The model is trained, evaluated, and visualized using Google Colab or Jupyter Notebook.

---

## 📌 Features

- Feed-forward neural network (784 → 128 → 64 → 10)
- Trained on the classic MNIST dataset
- Achieves over **97% accuracy** on the test set
- Includes data visualization and class probability plots
- Built using PyTorch and Torchvision
- Ready-to-run in **Google Colab**

---

## 📂 Project Structure

MNIST_Classifier.ipynb # Main notebook with full training + testing

Requirements.txt # Required packages (for local execution)

README.md # Project documentation
---

## 🚀 Getting Started

### 1. Clone the Repository
git clone https://github.com/yourusername/mnist-pytorch-classifier.git
cd mnist-pytorch-classifier

2. Install Dependencies (if running locally)
pip install -r requirements.txt
If you're using Google Colab, dependencies like torch, torchvision, and matplotlib are already installed.

📊 Model Architecture
The network has the following architecture:

Arduino
Input: 784 (28x28 flattened image)
↓
Linear (784 → 128) + ReLU
↓
Linear (128 → 64) + ReLU
↓
Linear (64 → 10) + LogSoftmax
Loss Function: Negative Log-Likelihood (NLLLoss)

Optimizer: Stochastic Gradient Descent (SGD) with Momentum

🧪 Results
Metric	Value
Test Accuracy	~97.2%
Epochs	15
Batch Size	64
Optimizer	SGD (lr=0.003, momentum=0.9)

📈 Visualization Example
The notebook includes visualization of:
Sample MNIST images
Class probabilities for predictions
Misclassifications and confidence levels

📌 Dataset
Source: MNIST (Yann LeCun)
60,000 training images and 10,000 test images
Grayscale images of handwritten digits (0–9)

📬 Contact
For questions, suggestions, or collaborations, feel free to open an issue or reach out via the repository.

📄 License
This project is released under the MIT License.

Let me know if you'd like:
- A version adapted for CNN instead of feed-forward
- A badge section (Python version, Colab badge, etc.)
- Screenshots or diagrams included

I'm happy to help further!
