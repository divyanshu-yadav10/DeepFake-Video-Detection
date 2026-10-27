# Spatio-Temporal Deepfake Video Detection (Xception + Bi-LSTM)







An end-to-end deep learning pipeline for detecting manipulated facial videos (Deepfakes) by combining spatial artifact extraction via an ImageNet-pretrained **Xception** Convolutional Neural Network with temporal sequence modeling using a **Bidirectional Long Short-Term Memory (Bi-LSTM)** network.

Spatial generative models often introduce subtle intra-frame facial blending artifacts alongside inter-frame temporal inconsistencies (such as unnatural blinking, frame-to-frame jitter, or lighting shifts). This project addresses both dimensions through a decoupled two-stage architecture: extracting `2048`-dimensional frame-level facial embeddings with Xception and classifying 10-frame temporal sequences with a regularized Bi-LSTM classifier—achieving **~94.3% training accuracy** and **~90.2% validation accuracy** across 100 epochs.

## Key Engineering Highlights

- **Automated Face Localization & Cropping**: Utilizes OpenCV Haar Cascade (`haarcascade_frontalface_default.xml`) to detect and isolate the largest facial region in each sampled video frame (`scaleFactor=1.3`, `minNeighbors=5`), eliminating background noise prior to spatial encoding.
- **Batched GPU Feature Extraction**: Leverages a frozen **Xception** backbone (`include_top=False` with `GlobalAveragePooling2D`) to transform `299 × 299 × 3` normalized facial crops into compact `2048`-dimensional feature vectors in GPU-accelerated batches (`BATCH_SIZE = 32`), serializing arrays into compressed `.npz` archives (`np.savez`) for rapid downstream training.
- **Bidirectional Temporal Modeling**: Captures both forward and backward temporal dependencies across uniform video frame windows (`num_frames = 10`) using a `Bidirectional(LSTM(16))` architecture with dropout regularization (`0.3`) and early stopping (`patience = 10` on validation loss).
- **Benchmark Dataset Compatibility**: Configured for evaluation and training on standard deepfake forensics benchmarks including **Celeb-DF** (`Celeb-real`, `Celeb-synthesis`, `test-real`, `test-fake`) and **FaceForensics++** (`c23` compressed sequences).

## System Architecture & Pipeline

| Pipeline Stage | Module / Script | Input | Processing Logic | Output |
| :--- | :--- | :--- | :--- | :--- |
| **1. Frame Sampling & Face Cropping** | [`feature_extraction.py`](https://drive.google.com/file/d/1z_Grel_rguIdH9yuRoh4LvX9lKi-30R-/view?usp=drivesdk) | Raw `.mp4` video | Uniformly samples $N=10$ frames across the video duration; detects largest face via Haar Cascade, crops, resizes to `299 × 299`, and normalizes pixels to `[0, 1]`. | Tensor of shape `(10, 299, 299, 3)` |
| **2. Spatial Feature Encoding** | [`feature_extraction.py`](https://drive.google.com/file/d/1z_Grel_rguIdH9yuRoh4LvX9lKi-30R-/view?usp=drivesdk) | Normalized face frames | Passes frames through ImageNet-pretrained **Xception** (`include_top=False`) followed by `GlobalAveragePooling2D`; serializes arrays with labels. | Feature archive `.npz` of shape `(num_videos, 10, 2048)` |
| **3. Temporal Sequence Training** | [`model_training.py`](https://drive.google.com/file/d/1R-rIwUeXd70RMKv4zx8iD143OJWBYmhD/view?usp=drivesdk) | `.npz` features + [`video_labels.csv`](https://drive.google.com/file/d/1HCrQV8mjqtn3P_B7ccQXypZtJqTh6tiu/view?usp=drivesdk) | Shuffles dataset, trains `Bidirectional(LSTM(16))` classifier with `Adam(lr=1e-5)` and `EarlyStopping(patience=10)`, and plots training/validation curves. | Trained model [`lstm_model.keras`](https://drive.google.com/file/d/1Qa7TPs3WqbRFfFKR2i9HzA0HQ-OS_Nev/view?usp=drivesdk) (`final_mix.keras`) |
| **4. End-to-End Video Inference** | [`prediction.py`](https://drive.google.com/file/d/1VtGMl2DnsfhqfQOdByNA7C3KKUJtRXGz/view?usp=drivesdk) | Unseen `.mp4` video | Extracts 10 face-cropped frames, computes Xception embeddings `(1, 10, 2048)`, and applies sigmoid threshold (`0.5`) via the trained Bi-LSTM model. | Binary prediction (`"Deepfake"` or `"Real"`) |

## Model Architecture & Hyperparameters

### Bi-LSTM Sequence Classifier ([`model_training.py`](https://drive.google.com/file/d/1R-rIwUeXd70RMKv4zx8iD143OJWBYmhD/view?usp=drivesdk))

| Layer | Type / Configuration | Output Shape | Details |
| :--- | :--- | :--- | :--- |
| **Input** | `Input(shape=(10, 2048))` | `(None, 10, 2048)` | 10 temporal steps × 2048 Xception spatial features |
| **Recurrent Encoder** | `Bidirectional(LSTM(16, return_sequences=False))` | `(None, 32)` | Concatenates 16 forward + 16 backward LSTM units |
| **Regularization** | `Dropout(0.3)` | `(None, 32)` | Mitigates overfitting on high-dimensional embeddings |
| **Dense Projection** | `Dense(8, activation='relu')` | `(None, 8)` | Non-linear feature compression |
| **Classification Head** | `Dense(1, activation='sigmoid')` | `(None, 1)` | Probability score $p \in [0, 1]$ (`> 0.5` = Deepfake) |

### Training Configuration

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Spatial Backbone** | `Xception (imagenet)` | Frozen weights (`trainable = False`), `299 × 299 × 3` input |
| **Optimizer** | `Adam` | Learning rate = `1e-5` (`0.00001`) |
| **Loss Function** | `binary_crossentropy` | Binary classification (`0` = Real, `1` = Deepfake) |
| **Batch Size** | `32` | Used for both Xception feature extraction and LSTM training |
| **Max Epochs** | `100` | Monitored with `EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)` |
| **Validation Split** | `0.20` | 80% training / 20% validation split after random index shuffling |

## Experimental Results & Performance

The Bidirectional LSTM classifier was trained over 100 epochs on the pre-extracted Xception spatial embeddings. The training and validation curves generated by `plot_training_history()` in [`model_training.py`](https://drive.google.com/file/d/1R-rIwUeXd70RMKv4zx8iD143OJWBYmhD/view?usp=drivesdk) are saved in `assets/result.png`:



### Quantitative Performance Summary & Training and Validation Accuracy and Loss 

![Training and Validation Accuracy and Loss](assets/result.png)

| Metric | Initial (Epoch 1) | Epoch 20 | Epoch 50 | Final (Epoch 100) |
| :--- | :--- | :--- | :--- | :--- |
| **Training Accuracy (%)** | ~71.0% | ~85.0% | ~90.0% | **~94.3%** |
| **Validation Accuracy (%)** | ~73.6% | ~85.5% | ~88.5% | **~90.2%** |
| **Training Loss (BCE)** | ~0.64 | ~0.41 | ~0.28 | **~0.18** |
| **Validation Loss (BCE)** | ~0.59 | ~0.39 | ~0.32 | **~0.28** |

- **Smooth Convergence**: The fine-grained Adam learning rate (`1e-5`) enables steady, monotonic loss reduction without gradient instability.
- **Strong Generalization**: Validation accuracy closely tracks training accuracy up to ~88% (Epoch 45) before plateauing at **~90.2%**, while `Dropout(0.3)` keeps the train-validation generalization gap well-bounded.

## Repository Structure

```text
DeepFake-Detection/
├── assets/
│   └── result.png               # Training & validation accuracy (%) and BCE loss curves (100 epochs)
├── feature_extraction.py        # OpenCV Haar Cascade face cropping + Xception feature extraction
├── model_training.py            # Bidirectional LSTM architecture, training loop, and Matplotlib plots
├── prediction.py                # End-to-end video inference pipeline using saved Keras weights
├── lstm_model.keras             # Pre-trained Bidirectional LSTM model weights (~64 MB)
├── video_labels.csv             # Ground-truth dataset manifest (149 test-real & test-fake video mappings)
├── requirements.txt             # Python package dependencies
├── .gitignore                   # Excludes raw video dataset directories from version control
└── README.md                    # Project documentation
```

## Getting Started

### 1. Clone the Repository & Set Up Environment

```bash
git clone https://github.com/divyanshu-yadav10/DeepFake-Video-Detection.git
cd DeepFake-Video-Detection

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # On Linux/macOS
# venv\Scripts\activate         # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare the Dataset

Organize your video dataset folders in the project root (excluded from Git via [`.gitignore`](https://drive.google.com/file/d/1XCqbmarAPQsiAadwGRPIc7v15j87fDPB/view?usp=drivesdk): `/Celeb-real`, `/Celeb-synthesis`, `/comDataset`, `/test-fake`, `/test-real`) and verify that your CSV label file matches the format in [`video_labels.csv`](https://drive.google.com/file/d/1HCrQV8mjqtn3P_B7ccQXypZtJqTh6tiu/view?usp=drivesdk):

```csv
file_name,label
test-real\00000.mp4,0
test-fake\id0_id1_0001.mp4,1
```

### 3. Extract Spatial Features with Xception

Run [`feature_extraction.py`](https://drive.google.com/file/d/1z_Grel_rguIdH9yuRoh4LvX9lKi-30R-/view?usp=drivesdk) to detect faces, sample 10 frames per video, and save the extracted `2048`-d Xception embeddings to `features_final.npz`:

```bash
python feature_extraction.py
```

### 4. Train the Bidirectional LSTM Classifier

Train the sequence model on the saved `.npz` embeddings and generate the accuracy/loss plot (`assets/result.png`):

```bash
python model_training.py
```

### 5. Run Inference on a Video

Use [`prediction.py`](https://drive.google.com/file/d/1VtGMl2DnsfhqfQOdByNA7C3KKUJtRXGz/view?usp=drivesdk) with the pre-trained model ([`lstm_model.keras`](https://drive.google.com/file/d/1Qa7TPs3WqbRFfFKR2i9HzA0HQ-OS_Nev/view?usp=drivesdk)) to classify a target `.mp4` video:

```bash
python prediction.py
```

## Tech Stack

- **Deep Learning**: [TensorFlow](https://www.tensorflow.org/) & [Keras](https://keras.io/) (`Xception`, `Bidirectional`, `LSTM`, `Dense`, `Dropout`, `EarlyStopping`)
- **Computer Vision**: [OpenCV](https://opencv.org/) (`cv2.VideoCapture`, `cv2.CascadeClassifier` Haar Cascades)
- **Data Processing**: [NumPy](https://numpy.org/) & [Pandas](https://pandas.pydata.org/)
- **Visualization**: [Matplotlib](https://matplotlib.org/)
