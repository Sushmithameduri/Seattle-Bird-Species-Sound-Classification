# 🐦 Sounds of Seattle Birds: A Neural Network Model for Bird Species Identification

## 📘 Project Overview
This project explores the application of **Convolutional Neural Networks (CNNs)** to identify bird species common to the Seattle area based on their vocalizations. Using **spectrograms** generated from bird call recordings, both **binary** and **multi-class classification models** were trained to predict species. The system was further tested on **external audio clips** to evaluate generalization.

---

## 📑 Table of Contents
1. [Introduction](#-introduction)
2. [Dataset](#-dataset)
3. [Data Preprocessing](#-data-preprocessing)
4. [Model Architecture](#-model-architecture)
   - [Binary Model](#-binary-model)
   - [Multi-class Model](#-multi-class-model)
5. [Hyperparameter Tuning](#-hyperparameter-tuning)
6. [Evaluation and Results](#-evaluation-and-results)
7. [External Test Predictions](#-external-test-predictions)
8. [Discussion](#-discussion)
9. [Dependencies](#-dependencies)
10. [Usage](#-usage)
11. [Contact](#-contact)

---

## 🧠 Introduction
Traditional bird call identification methods are manual, time-consuming, and prone to human error. This project leverages **deep learning**—specifically **CNNs**—to automate this process, classifying bird species directly from spectrograms (time-frequency visualizations of audio signals).

Two models were developed:
- **Binary Classifier** – Distinguishes between *Black-capped Chickadee* and *Dark-eyed Junco*.
- **Multi-class Classifier** – Classifies among 12 Seattle bird species.

---

## 🎧 Dataset

**Source:**  
- [Xeno-Canto Bird Recordings Extended (A–M)](https://www.kaggle.com/datasets/rohanrao/xeno-canto-bird-recordings-extended-a-m)
- Training data on spectrograms of 10 mp3 sound clips of various lengths for each of 12 bird species [Spectrograms.h5](https://drive.google.com/file/d/1Fb9rIEbgg-eJGfzbiDqBsOP98Mvf0N0L/view?usp=sharing) 
- Test data on 3 unlabeled mystery bird call clips are available in the repo named as [test_birds.zip](https://github.com/Sushmithameduri/Seattle-Bird-Species-Sound-Classification/blob/main/test_birds.zip)

**Provided via:** Seattle University Canvas platform  
**Data format:** HDF5 (`.h5`) spectrograms derived from 10 sound clips per species.  
**Sampling rate:** 22,050 Hz (half the original).  
**Spectrogram size:** 343 (time) × 256 (frequency) pixels.

**Included Species (12):**
1. American Crow *(amecro)*
2. Barn Swallow *(barswa)*
3. Black-capped Chickadee *(bkcchi)*
4. Blue Jay *(blujay)*
5. Dark-eyed Junco *(daejun)*
6. House Finch *(houfin)*
7. Mallard *(mallar3)*
8. Northern Flicker *(norfli)*
9. Red-winged Blackbird *(rewbla)*
10. Steller’s Jay *(stejay)*
11. Western Meadowlark *(wesmea)*
12. White-crowned Sparrow *(whcspa)*

---

## ⚙️ Data Preprocessing

- Audio recordings were split into **2-second windows** where bird calls were detected.
- Each segment was transformed into a **spectrogram** using `librosa`.
- Spectrograms were **normalized** (0–1 range) and reshaped for CNN input:  
  `(height=256, width=343, channels=1)`.
- **HDF5 file format** (`spectrograms.h5`) was used for efficient storage and retrieval.

---

## 🧩 Model Architecture

### 🐤 Binary Model
**Goal:** Classify between *Black-capped Chickadee* (0) and *Dark-eyed Junco* (1)

**Architecture:**
- Conv2D (32 filters, 3×3) + ReLU  
- MaxPooling2D (2×2)  
- Conv2D (64 filters, 3×3) + ReLU  
- MaxPooling2D (2×2)  
- Flatten + Dense (128, ReLU) + Dropout (0.5)  
- Output Dense (1, Sigmoid)

**Training Parameters:**
- Optimizer: **RMSprop**
- Loss: **Binary Crossentropy**
- Epochs: 10
- Batch Size: 64
- Validation Split: 0.2

**Accuracy:**  
- Initial Model: **86.96%**  
- Tuned Model: **94.73%**

---

### 🕊️ Multi-class Model
**Goal:** Classify among all 12 bird species.

**Architecture:**
- Similar CNN structure as binary model  
- Final Layer: Dense(12, Softmax)  
- Loss: **Categorical Crossentropy**  
- Epochs: 10  
- Batch Size: 16  
- Validation Split: 0.2

**Accuracy:**  
- Initial Model: **70.69%**  
- Tuned Model: **75.26%**

---

## 🔧 Hyperparameter Tuning

Explored combinations of:
- **Batch size:** 8, 16, 32  
- **Filters:** 8, 16, 32  
- **Dense units:** 16, 32  
- **Dropout rate:** 0.3, 0.5  

Best results achieved with:
- Batch size: 16  
- Dropout: 0.5  
- Filters: 32  
- Optimizer: RMSprop  

---

## 📊 Evaluation and Results

| Model Type | Validation Accuracy | Test Accuracy | Training Time |
|-------------|--------------------|----------------|----------------|
| Binary (Initial) | 86.96% | ~87% | 3.5 min |
| Binary (Tuned) | **94.73%** | 94% | 20 min |
| Multi-class (Initial) | 70.69% | ~71% | 6 min |
| Multi-class (Tuned) | **75.26%** | 75% | ~1 hr |

---

##  External Test Predictions

| Audio File | Predicted Species | Multiple Birds? | Reasoning |
|-------------|------------------|-----------------|------------|
| test1.mp3 | Dark-eyed Junco | No | Clear, distinct call pattern |
| test2.mp3 | Northern Flicker | **Yes** | Overlapping calls at varying frequencies |
| test3.mp3 | Dark-eyed Junco | No | Consistent pattern, no overlap |

---

## 💬 Discussion

- CNNs effectively learned to classify bird species based on call spectrograms.
- **Binary classification** was easier due to clear differences between species.
- **Multi-class model** faced challenges due to overlapping frequency patterns.
- Frequent confusions:
  - *Dark-eyed Junco* ↔ *Black-capped Chickadee*
  - *American Crow* ↔ *Red-winged Blackbird*
  - *Barn Swallow* ↔ *Dark-eyed Junco*
- Training time and hardware constraints were key limitations.
- Some test clips (notably `test2.mp3`) suggested multiple bird calls.

**Alternative models:**  
Support Vector Machines (SVMs), Random Forests, and Recurrent Neural Networks (RNNs) could also be explored for better temporal pattern analysis.

---


## 🧩 Dependencies

Install required packages using:
```bash
pip install numpy pandas tensorflow keras matplotlib librosa h5py scikit-learn
```

#### Key Libraries Used:
 * TensorFlow / Keras – model creation and training

* librosa – audio analysis and spectrogram generation

* h5py – HDF5 data handling

* matplotlib – data visualization

## 🚀 Usage

1. Prepare data

   * Download the HDF5 spectrogram dataset.
   *  Place in working directory as spectrograms.h5.

2. Clone Repository
   ```bash
   git clone https://github.com/Sushmithameduri/Seattle-Bird-Species-Sound-Classification
   cd Sushmithameduri/Seattle-Bird-Species-Sound-Classification
   ```
3. Run the notebook or script
   *Open "Bird_Sound_Prediction_Code.ipynb" in Google Colab or Jupyter.
   ```bash
   python Birds_Sound_Prediction_Code.iynb
   ```
   * Update file paths to point to spectrograms.h5 https://drive.google.com/file/d/1Fb9rIEbgg-eJGfzbiDqBsOP98Mvf0N0L/view?usp=sharing.
   * Execute cells to train models and generate plots.

4. Predict External Audio
   * Place mp3 files in the /test_birds/ folder.
   * Run the inference section to classify each clip.

5. View results
   * Predicted species will print to console and optionally save to CSV.


## 📈 Insights & Learnings

* CNNs are effective for audio-to-image transformations like spectrograms.
* Model interpretability through confusion matrices and spectrogram analysis helps identify misclassification causes.
* Hardware limitations affect training time — model tuning should balance performance with efficiency.
* Future enhancements: multi-label classification, transfer learning (ResNet / VGG), or integrating RNN + CNN hybrids.

## 📬 Contact

### Sushmitha Meduri
1. 📧 Email: sivasushmithameduri@gmail.com
2. 🔗 LinkedIn: https://www.linkedin.com/in/sushmitha-meduri/
3. 💻 GitHub: https://github.com/Sushmithameduri/
