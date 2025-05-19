
# CREMA-D Emotion Recognition System

This repository contains a deep learning system for speech emotion recognition using the CREMA-D dataset. The system uses mel spectrograms extracted from audio files and a CNN-LSTM architecture with residual connections to classify emotions.

## Overview

The system consists of three main components:

1. **Mel Spectrogram Generator** (`mel_generator.py`): Processes audio files to create mel spectrograms with optional data augmentation.
2. **Model Training** (`cremad_custom2.py`): Implements and trains the CNN-LSTM model with residual connections.
3. **Model Testing** (`custom2_tester.py`): Tests the trained model on new audio files and provides detailed analysis.

## Requirements

- Python 3.6+
- PyTorch
- NumPy
- Librosa
- Scikit-learn
- Matplotlib
- Seaborn
- tqdm
- Pandas
- Tabulate (for testing script)
- SoundFile


## Mel Spectrogram Generator

The `mel_generator.py` script processes audio files from the CREMA-D dataset and creates mel spectrograms with optional data augmentation.

### Features:

- Extracts emotion labels from filenames
- Creates mel spectrograms and saves them as .npy files
- Applies various augmentation techniques:
    - Time shifting (positive and negative)
    - Pitch shifting (up and down)
    - Time stretching (faster and slower)
    - Adding random noise (low and medium levels)


### Usage:

```bash
python mel_generator.py --input_dir /path/to/audio/files --output_dir /path/to/save/spectrograms [--no_augment] [--n_augmentations 3]
```


## Model Training

The `cremad_custom2.py` script implements and trains the CNN-LSTM model with residual connections.

### Model Architecture:

- **CNN Blocks**: Four CNN blocks with residual connections and batch normalization
- **LSTM**: Bidirectional LSTM with 512 hidden units and 2 layers
- **Attention**: Multi-head self-attention mechanism
- **Fully Connected**: Three fully connected layers with batch normalization


### Training Features:

- Weighted random sampling to handle class imbalance
- Mixed precision training for faster processing
- Learning rate scheduler (ReduceLROnPlateau)
- Early stopping to prevent overfitting
- Comprehensive evaluation metrics and visualizations


### Usage:

```bash
python cremad_custom2.py
```


## Model Testing

The `custom2_tester.py` script tests the trained model on new audio files and provides detailed analysis.

### Features:

- Processes WAV files to extract mel spectrograms
- Classifies emotions using the trained model
- Provides detailed confidence scores for each prediction
- Generates visualizations of mel spectrograms and confidence distributions
- Saves results in CSV format


### Usage:

```bash
python custom2_tester.py --input_dir /path/to/wav/files --model_path /path/to/model.pth --output_dir /path/to/save/results [--max_width 191] [--cpu] [--visualize]
```


## Emotion Classes

The system classifies audio into six emotion categories:

1. Angry (ANG)
2. Disgust (DIS)
3. Fear (FEA)
4. Happy (HAP)
5. Neutral (NEU)
6. Sad (SAD)

## Implementation Details

### Data Preprocessing:

- Audio files are converted to mel spectrograms
- Spectrograms are normalized and padded/truncated to a uniform size
- Data augmentation is applied to increase the training set size


### Training Process:

- The dataset is split into training, validation, and test sets
- Class weights are applied to handle imbalanced classes
- Training uses cross-entropy loss and AdamW optimizer
- Early stopping is implemented based on validation accuracy


### Evaluation:

- Confusion matrix and classification report are generated
- Training and validation metrics are plotted
- Best model is saved based on validation accuracy


## Notes

- The model implementation uses PyTorch and can run on both CPU and GPU
- The default mel spectrogram width is 191 frames, but this can be adjusted
- The system is designed to handle class imbalance in the dataset


[^1]: cremad_custom2.py

[^2]: custom2_tester.py

[^3]: mel_generator.py

