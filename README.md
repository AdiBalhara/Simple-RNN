# Simple RNN - IMDB Sentiment Analysis

A deep learning project implementing a Simple Recurrent Neural Network (RNN) for sentiment analysis on the IMDB movie reviews dataset using TensorFlow/Keras.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-orange)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-RNN-green)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Architecture](#project-architecture)
- [Model Architecture](#model-architecture)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Technologies Used](#technologies-used)

---

## 🎯 Overview

This project demonstrates the implementation of a Simple Recurrent Neural Network for binary sentiment classification (positive/negative) on the IMDB movie reviews dataset. The project includes:

- **Word Embedding techniques** for text representation
- **SimpleRNN architecture** for sequence processing
- **Early Stopping** for preventing overfitting
- Pre-trained model saved for inference

---

## 🏗️ Project Architecture

```
Simple RNN/
│
├── SimpleRNN.ipynb          # Main notebook - RNN model training & evaluation
├── embedding.ipynb          # Word embedding demonstration notebook
├── simple_rnn_imdb.h5       # Pre-trained model weights
├── requirement.txt          # Project dependencies
├── README.md                # Project documentation
└── Senv/                    # Conda virtual environment
```

---

## 🧠 Model Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT LAYER                               │
│              (Sequence of word indices)                      │
│                  Shape: (500,)                               │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  EMBEDDING LAYER                             │
│         Vocabulary Size: 10,000 | Embedding Dim: 128         │
│              Output Shape: (500, 128)                        │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   SimpleRNN LAYER                            │
│              Units: 128 | Activation: ReLU                   │
│                 Output Shape: (128,)                         │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    DENSE LAYER                               │
│            Units: 1 | Activation: Sigmoid                    │
│               Output: Binary Classification                  │
│              (Positive: 1, Negative: 0)                      │
└─────────────────────────────────────────────────────────────┘
```

### Model Summary

| Layer          | Output Shape | Parameters |
|----------------|--------------|------------|
| Embedding      | (None, 500, 128) | 1,280,000 |
| SimpleRNN      | (None, 128)      | 32,896    |
| Dense          | (None, 1)        | 129       |
| **Total**      |                  | **1,313,025** |

---

## ✨ Features

- **Text Preprocessing**: Sequence padding to uniform length (500 tokens)
- **One-Hot Encoding**: Word to integer mapping using vocabulary of 10,000 words
- **Word Embeddings**: Dense vector representation of words (128 dimensions)
- **Early Stopping**: Automatic training termination when validation loss stops improving
- **Binary Classification**: Sentiment prediction (Positive/Negative)

---

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Simple-RNN.git
   cd Simple-RNN
   ```

2. **Create a virtual environment**
   ```bash
   conda create -n rnn_env python=3.13
   conda activate rnn_env
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirement.txt
   ```

---

## 🚀 Usage

### Training the Model

```python
# Open SimpleRNN.ipynb and run all cells
# The model will:
# 1. Load IMDB dataset (25,000 training, 25,000 testing samples)
# 2. Preprocess and pad sequences
# 3. Train with early stopping
# 4. Save the model as 'simple_rnn_imdb.h5'
```

### Loading Pre-trained Model

```python
from tensorflow.keras.models import load_model

model = load_model('simple_rnn_imdb.h5')
```

### Making Predictions

```python
# Preprocess your review text and predict
prediction = model.predict(preprocessed_review)
sentiment = "Positive" if prediction > 0.5 else "Negative"
```

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Dataset | IMDB Movie Reviews |
| Training Samples | 25,000 |
| Test Samples | 25,000 |
| Vocabulary Size | 10,000 |
| Max Sequence Length | 500 |
| Optimizer | Adam |
| Loss Function | Binary Crossentropy |

---

## 🔧 Technologies Used

- **Python 3.13**
- **TensorFlow 2.20.0** - Deep learning framework
- **Keras** - High-level neural network API
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **Scikit-learn** - Machine learning utilities
- **Matplotlib** - Data visualization
- **Streamlit** - Web application framework

---

## 📝 Resume Description

> **Simple RNN - IMDB Sentiment Analysis**

• Built a deep learning sentiment analysis model using Simple RNN to classify IMDB movie reviews as positive or negative  
• Implemented complete NLP pipeline: Text preprocessing, One-hot encoding, Sequence padding, Word embeddings & Model training  
• Designed neural network architecture with Embedding layer (10,000 vocab, 128-dim), SimpleRNN layer (128 units), and Dense output layer  
• Applied Early Stopping callback to prevent overfitting by monitoring validation loss  
• Utilized pad_sequences for uniform input length (500 tokens) to handle variable-length text data  
• Trained on 50,000 IMDB reviews (25K training, 25K testing) achieving binary sentiment classification  
• Implemented word-to-index mapping and reverse decoding for model interpretability  
• **Tech Stack**: Python, TensorFlow, Keras, NumPy, Pandas, Scikit-learn, Matplotlib, Streamlit

### Alternative Shorter Version:

> Built a sentiment analysis model using Simple RNN architecture for IMDB movie review classification. Implemented word embeddings and sequence processing with TensorFlow/Keras, achieving binary sentiment prediction on 50,000 reviews. **Skills**: Python, TensorFlow, Keras, RNN, NLP, Deep Learning.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

---

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)

---

⭐ Star this repository if you found it helpful!
