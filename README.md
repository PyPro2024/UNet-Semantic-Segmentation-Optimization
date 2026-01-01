# Semantic Segmentation with U-Net

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?style=flat&logo=keras)
![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat&logo=python)

## 📌 Project Overview
This project implements the **U-Net** architecture, a specialized Convolutional Neural Network (CNN) designed for biomedical image segmentation, applied here to the task of **Human Semantic Segmentation**.

The goal is to generate a pixel-perfect mask that separates the person (foreground) from the background. The project involves:
1.  **Synthetic Data Prototyping:** Validating the pipeline on generated geometric shapes.
2.  **Real-World Application:** Training on the "Human Segmentation Dataset" using a full Encoder-Decoder pipeline.
3.  **Hyperparameter Tuning:** Systematically experimenting with network depth and learning rates to improve segmentation accuracy.

## 🧠 Model Architecture: U-Net
The model follows the classic "U" shape architecture:



* **Encoder (Contracting Path):** Captures context via convolutional and max-pooling layers.
* **Decoder (Expansive Path):** Enables precise localization using upsampling and **Skip Connections**.
* **Skip Connections:** Concatenate feature maps from the encoder to the decoder to recover spatial information lost during downsampling.

## 🔬 Experiments & Optimization
I conducted a comparative analysis to find the optimal configuration:

| Experiment | Configuration | Val Loss | Val Accuracy | Conclusion |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | Standard U-Net | 0.5106 | 76.68% | Good starting point. |
| **Deeper Net** | **+1 Layer in Encoder/Decoder** | **0.4289** | **81.67%** | **Best Performance.** Deeper features captured complex human boundaries better. |
| **High LR** | Learning Rate = 0.001 | 0.4723 | 78.46% | Performance degraded; optimization was too aggressive. |

## 🛠️ Tech Stack
* **Deep Learning:** TensorFlow, Keras (Functional API)
* **Image Processing:** OpenCV, NumPy
* **Dataset:** Human Segmentation Dataset (Cloned from GitHub)
* **Visualization:** Matplotlib (for Overlaying Masks)

## 🚀 How to Run
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR-USERNAME/UNet-Semantic-Segmentation-Keras.git](https://github.com/YOUR-USERNAME/UNet-Semantic-Segmentation-Keras.git)
    ```
2.  **Install dependencies:**
    ```bash
    pip install tensorflow numpy opencv-python matplotlib sklearn
    ```
3.  **Run the Notebook:**
    Open `UNet_Segmentation.ipynb` in Jupyter Notebook or Google Colab.

## 🖼️ Results
The optimized model successfully predicts segmentation masks on unseen test images, handling complex boundaries around hair and clothing.

*(Note: Visual results comparing Ground Truth vs. Predicted Masks are available in the notebook.)*

---
*If you find this segmentation project helpful, feel free to ⭐ the repo!*
