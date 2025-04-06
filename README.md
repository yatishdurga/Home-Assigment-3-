# Home-Assigment-3-



# Deep Learning Projects - Autoencoders and RNNs

This repository contains implementations of several deep learning tasks using TensorFlow and Keras. The focus is on unsupervised and supervised learning models like **Autoencoders**, **Denoising Autoencoders**, **Recurrent Neural Networks (RNNs)**, and **Sentiment Analysis using LSTM**.

---

## 📌 Q1: Implementing a Basic Autoencoder

### 🧠 Task
Build a fully connected autoencoder and evaluate its performance on reconstructing the MNIST handwritten digits.

### 📝 Steps:
1. Load the MNIST dataset using `tensorflow.keras.datasets`.
2. Define a fully connected (Dense) autoencoder:
   - **Encoder**: Input layer (784), hidden layer (32).
   - **Decoder**: Hidden layer (32), output layer (784).
3. Compile and train the autoencoder using **binary cross-entropy** loss.
4. Visualize original vs. reconstructed images after training.
5. Modify the latent space (e.g., 16, 64) and analyze reconstruction quality.

### 💡 Hint
Use `Model()` from `tensorflow.keras.models` and `Dense()` layers.

---

## 📌 Q2: Implementing a Denoising Autoencoder

### 🧠 Task
Train an autoencoder to remove noise from images, using the MNIST dataset.

### 📝 Steps:
1. Add **Gaussian noise** (mean=0, std=0.5) to input images using `np.random.normal()`.
2. Output during training remains the **clean image**.
3. Train the model and visualize **noisy vs. reconstructed** images.
4. Compare reconstruction quality between:
   - Basic Autoencoder
   - Denoising Autoencoder
5. Explain a real-world application of denoising autoencoders (e.g., medical imaging, security systems).

---

## 📌 Q3: Implementing an RNN for Text Generation

### 🧠 Task
Train an LSTM-based Recurrent Neural Network to predict the next character in a sequence and generate text.

### 📝 Steps:
1. Load a text dataset (e.g., *The Little Prince*, *Shakespeare Sonnets*).
2. Convert text into a sequence of characters using **one-hot encoding** or **embeddings**.
3. Define an **LSTM-based RNN model** using `tensorflow.keras.layers.LSTM`.
4. Train the model and generate new text by sampling one character at a time.
5. Explain **temperature scaling** and its effect on randomness and diversity of generated text.

---

## 📌 Q4: Sentiment Classification Using RNN

### 🧠 Task
Perform sentiment analysis on movie reviews using the IMDB dataset and an LSTM-based model.

### 📝 Steps:
1. Load the **IMDB dataset** using `tensorflow.keras.datasets.imdb`.
2. Preprocess the data:
   - Tokenization
   - Padding sequences to equal length
3. Define and train an **LSTM-based classifier**.
4. Evaluate performance using:
   - **Confusion matrix**
   - **Classification report** (accuracy, precision, recall, F1-score)
5. Discuss the **precision-recall tradeoff** and its importance in NLP classification tasks.

### 💡 Hint
Use `confusion_matrix` and `classification_report` from `sklearn.metrics`.



