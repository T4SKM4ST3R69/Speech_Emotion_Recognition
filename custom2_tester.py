import os
import numpy as np
import torch
import librosa
import soundfile as sf
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from tabulate import tabulate
from torch import nn


# CNN-LSTM model with residual connections and batch normalization
class ImprovedCNNLSTMModel(nn.Module):
    def __init__(self, num_classes=6, input_channels=1, input_height=128, input_width=191):
        super(ImprovedCNNLSTMModel, self).__init__()

        # Store input dimensions
        self.input_height = input_height
        self.input_width = input_width

        # Activation functions
        self.relu = nn.ReLU()

        # First CNN block
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # Second CNN block with residual connection
        self.conv2a = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2a = nn.BatchNorm2d(128)
        self.conv2b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn2b = nn.BatchNorm2d(128)
        self.conv2_shortcut = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # Third CNN block with residual connection
        self.conv3a = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3a = nn.BatchNorm2d(256)
        self.conv3b = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn3b = nn.BatchNorm2d(256)
        self.conv3_shortcut = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        # Fourth CNN block with residual connection
        self.conv4a = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4a = nn.BatchNorm2d(512)
        self.conv4b = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn4b = nn.BatchNorm2d(512)
        self.conv4_shortcut = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        # Dropout for regularization
        self.dropout_cnn = nn.Dropout(0.4)
        self.dropout_lstm = nn.Dropout(0.4)

        # Calculate output dimensions after CNN
        self._calculate_cnn_output_dims()

        # Bidirectional LSTM with increased hidden size
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.4
        )

        # Multi-head self-attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=1024, num_heads=8)

        # Fully connected layers with batch normalization
        self.fc1 = nn.Linear(1024, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, num_classes)

    def _calculate_cnn_output_dims(self):
        # Create a dummy input to calculate output dimensions
        x = torch.zeros(1, 1, self.input_height, self.input_width)

        # Apply CNN layers
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))

        # Block 2 with residual
        identity = self.conv2_shortcut(x)
        x = self.relu(self.bn2a(self.conv2a(x)))
        x = self.bn2b(self.conv2b(x))
        x = self.relu(x + identity)
        x = self.pool2(x)

        # Block 3 with residual
        identity = self.conv3_shortcut(x)
        x = self.relu(self.bn3a(self.conv3a(x)))
        x = self.bn3b(self.conv3b(x))
        x = self.relu(x + identity)
        x = self.pool3(x)

        # Block 4 with residual
        identity = self.conv4_shortcut(x)
        x = self.relu(self.bn4a(self.conv4a(x)))
        x = self.bn4b(self.conv4b(x))
        x = self.relu(x + identity)
        x = self.pool4(x)

        # Get output dimensions
        self.cnn_output_height = x.size(2)
        self.cnn_output_width = x.size(3)
        self.lstm_input_size = 512 * self.cnn_output_width

    def forward(self, x):
        batch_size = x.size(0)

        # Block 1
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        # Block 2 with residual
        identity = self.conv2_shortcut(x)
        x = self.relu(self.bn2a(self.conv2a(x)))
        x = self.bn2b(self.conv2b(x))
        x = self.relu(x + identity)
        x = self.pool2(x)

        # Block 3 with residual
        identity = self.conv3_shortcut(x)
        x = self.relu(self.bn3a(self.conv3a(x)))
        x = self.bn3b(self.conv3b(x))
        x = self.relu(x + identity)
        x = self.pool3(x)

        # Block 4 with residual
        identity = self.conv4_shortcut(x)
        x = self.relu(self.bn4a(self.conv4a(x)))
        x = self.bn4b(self.conv4b(x))
        x = self.relu(x + identity)
        x = self.pool4(x)

        x = self.dropout_cnn(x)

        # Reshape for LSTM
        time_steps = x.size(2)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(batch_size, time_steps, -1)

        # LSTM processing
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out.permute(1, 0, 2)
        attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_output = attn_output.permute(1, 0, 2)

        # Global average pooling over time dimension
        x = torch.mean(attn_output, dim=1)

        # Fully connected layers with batch normalization
        x = self.dropout_lstm(x)
        x = self.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_lstm(x)
        x = self.relu(self.bn_fc2(self.fc2(x)))
        x = self.fc3(x)

        return x


# Extract mel spectrogram from audio file with same parameters as training
def extract_mel_spectrogram(audio_path, max_width=191, sr=16000, n_fft=2048, hop_length=512, n_mels=128):
    try:
        # Load audio file
        audio, sample_rate = librosa.load(audio_path, sr=sr)

        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )

        # Convert to log scale (dB)
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize the mel spectrogram (exactly as in the training code)
        mel_spec = (mel_spec - np.mean(mel_spec)) / (np.std(mel_spec) + 1e-8)

        # Pad or resize to ensure uniform dimensions
        if mel_spec.shape[1] < max_width:
            # Pad with zeros to match max_width
            padded_spec = np.zeros((mel_spec.shape[0], max_width))
            padded_spec[:, :mel_spec.shape[1]] = mel_spec
            mel_spec = padded_spec
        elif mel_spec.shape[1] > max_width:
            # Truncate to max_width
            mel_spec = mel_spec[:, :max_width]

        return mel_spec

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        # Return a dummy spectrogram in case of error
        return np.zeros((n_mels, max_width))

# Visualize and save the mel spectrogram
def visualize_mel_spectrogram(mel_spec, output_path):
    plt.figure(figsize=(10, 4))

    # Create a custom colormap that mimics librosa's default
    colors = [(1, 1, 1), (0, 0, 0.8)]  # White to blue
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=100)

    plt.imshow(mel_spec, aspect='auto', origin='lower', cmap=cmap)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.ylabel('Mel Frequency')
    plt.xlabel('Time Frame')
    plt.tight_layout()

    plt.savefig(output_path)
    plt.close()

# Load the trained model
def load_model(model_path, input_height=128, input_width=191, device='cuda'):
    model = ImprovedCNNLSTMModel(
        num_classes=6,
        input_channels=1,
        input_height=input_height,
        input_width=input_width
    ).to(device)

    # Load model weights
    if device == 'cuda':
        checkpoint = torch.load(model_path)
    else:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    # Check if we need to extract state_dict from checkpoint
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model

# Classify a single audio sample
def classify_audio(model, mel_spec, device='cuda'):
    # Add batch and channel dimensions
    mel_spec = np.expand_dims(mel_spec, axis=0)  # Add batch dimension
    mel_spec = np.expand_dims(mel_spec, axis=0)  # Add channel dimension

    # Convert to torch tensor
    mel_spec_tensor = torch.FloatTensor(mel_spec).to(device)

    # Get prediction
    with torch.no_grad():
        outputs = model(mel_spec_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(outputs, dim=1).item()

    return pred_class, probabilities.cpu().numpy()[0]

# Process all WAV files in a directory
def process_directory(input_dir, model_path, output_dir, device='cuda', max_width=191, visualize=False):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define emotion mapping
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad']

    # Load model
    print(f"Loading model from {model_path}...")
    model = load_model(model_path, input_width=max_width, device=device)

    # Initialize results storage
    results = []

    # Get all WAV files
    wav_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.wav')]
    print(f"Found {len(wav_files)} WAV files to process")

    # Process each WAV file
    for wav_file in tqdm(wav_files, desc="Processing audio files"):
        try:
            file_path = os.path.join(input_dir, wav_file)

            # Extract mel spectrogram
            mel_spec = extract_mel_spectrogram(file_path, max_width=max_width)

            # Visualize spectrogram if requested
            if visualize:
                vis_path = os.path.join(output_dir, f"{os.path.splitext(wav_file)[0]}_mel.png")
                visualize_mel_spectrogram(mel_spec, vis_path)

            # Classify audio
            pred_class, probabilities = classify_audio(model, mel_spec, device=device)

            # Prepare result
            result = {
                'file': wav_file,
                'predicted_class': emotion_labels[pred_class],
                'predicted_class_id': pred_class
            }

            # Add probabilities for each emotion
            for i, emotion in enumerate(emotion_labels):
                result[f'{emotion}_confidence'] = float(probabilities[i])

            results.append(result)

        except Exception as e:
            print(f"Error processing {wav_file}: {e}")

    # Convert results to DataFrame
    df = pd.DataFrame(results)

    # Save results as CSV
    csv_path = os.path.join(output_dir, 'classification_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    # Print detailed confidence scores for each file
    print("\nDetailed Confidence Scores for Each File:")

    # Format probabilities as percentages with 2 decimal places
    formatted_df = df.copy()
    confidence_columns = [col for col in df.columns if col.endswith('_confidence')]

    # Format the confidence values as percentages
    for col in confidence_columns:
        formatted_df[col] = formatted_df[col].apply(lambda x: f"{x * 100:.2f}%")

    # Rename columns for better readability
    column_mapping = {f'{emotion}_confidence': emotion for emotion in emotion_labels}
    formatted_df = formatted_df.rename(columns=column_mapping)

    # Select relevant columns for detailed display
    display_columns = ['file', 'predicted_class'] + emotion_labels

    # Print each file with its confidence scores
    for idx, row in formatted_df[display_columns].iterrows():
        print(f"\nFile: {row['file']}")
        print(f"Predicted Emotion: {row['predicted_class']}")
        print("Confidence Scores:")
        for emotion in emotion_labels:
            print(f"  - {emotion}: {row[emotion]}")
        print("-" * 50)

    # Print summary table
    print("\nClassification Results Summary:")
    summary_table = []

    # Count occurrences of each predicted class
    class_counts = df['predicted_class'].value_counts().to_dict()
    total_samples = len(df)

    for emotion in emotion_labels:
        count = class_counts.get(emotion, 0)
        percentage = (count / total_samples) * 100 if total_samples > 0 else 0
        summary_table.append([emotion, count, f"{percentage:.2f}%"])

    print(tabulate(summary_table, headers=["Emotion", "Count", "Percentage"], tablefmt="grid"))

    # Create and save a visualization of confidence score distributions
    confidence_fig_path = os.path.join(output_dir, 'confidence_distribution.png')
    visualize_confidence_distribution(df, emotion_labels, confidence_fig_path)
    print(f"Confidence distribution visualization saved to {confidence_fig_path}")

    return df

# Create a visualization of confidence score distributions for each emotion
def visualize_confidence_distribution(df, emotion_labels, output_path):
    plt.figure(figsize=(12, 8))

    # Get confidence columns
    confidence_columns = [f'{emotion}_confidence' for emotion in emotion_labels]

    # Create box plots
    boxplot_data = [df[col].values for col in confidence_columns]
    plt.boxplot(boxplot_data, labels=emotion_labels)

    # Add individual points for better visibility of the distribution
    for i, col in enumerate(confidence_columns):
        # Add a small jitter to x-position to avoid overlapping points
        x = np.random.normal(i + 1, 0.04, size=len(df))
        plt.scatter(x, df[col], alpha=0.3, s=10)

    plt.title('Distribution of Confidence Scores by Emotion')
    plt.ylabel('Confidence Score')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig(output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Test CREMA-D emotion classification model on WAV files.')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing WAV files')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='classification_results', help='Directory to save results')
    parser.add_argument('--max_width', type=int, default=191, help='Maximum width of mel spectrogram')
    parser.add_argument('--cpu', action='store_true', help='Use CPU instead of GPU')
    parser.add_argument('--visualize', action='store_true', help='Visualize mel spectrograms')

    args = parser.parse_args()

    # Set device
    device = 'cpu' if args.cpu or not torch.cuda.is_available() else 'cuda'
    print(f"Using device: {device}")

    # Process the directory
    process_directory(
        input_dir=args.input_dir,
        model_path=args.model_path,
        output_dir=args.output_dir,
        device=device,
        max_width=args.max_width,
        visualize=args.visualize
    )


if __name__ == "__main__":
    main()