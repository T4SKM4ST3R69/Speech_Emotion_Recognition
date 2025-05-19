import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data.sampler import WeightedRandomSampler
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Define the dataset class for CREMA-D
class CREMADDataset(Dataset):
    def __init__(self, mel_specs_paths=None, labels=None, base_dir=None, transform=None, max_width=None):
        self.transform = transform
        self.max_width = max_width

        if mel_specs_paths is not None and labels is not None:
            # Direct initialization with paths and labels
            self.samples = list(zip(mel_specs_paths, labels))
        elif base_dir is not None:
            
            self.samples = []
            self._build_dataset(base_dir)
        else:
            raise ValueError("Either provide mel_specs_paths and labels OR base_dir")

    def _build_dataset(self, base_dir):
        # Define emotion mapping
        emotion_to_label = {
            'ANG': 0, 'DIS': 1, 'FEA': 2, 'HAP': 3, 'NEU': 4, 'SAD': 5
        }

        # Count samples per class for reporting
        class_counts = {emotion: 0 for emotion in emotion_to_label.keys()}

        # Process each emotion folder
        for emotion_folder in os.listdir(base_dir):
            emotion_path = os.path.join(base_dir, emotion_folder)

            # Skip if not a directory or not in our emotion mapping
            if not os.path.isdir(emotion_path) or emotion_folder not in emotion_to_label:
                continue

            emotion_label = emotion_to_label[emotion_folder]

            # Process all .npy files in the emotion folder
            for file in os.listdir(emotion_path):
                if file.endswith('.npy'):
                    mel_spec_path = os.path.join(emotion_path, file)
                    self.samples.append((mel_spec_path, emotion_label))
                    class_counts[emotion_folder] += 1

        print("Class distribution:")
        for emotion, count in class_counts.items():
            print(f"{emotion}: {count} samples")

        # Find max width if not provided
        if self.max_width is None:
            self._find_max_width()

    def _find_max_width(self):
        max_width = 0
        for path, _ in tqdm(self.samples, desc="Finding max spectrogram width"):
            try:
                mel_spec = np.load(path)
                max_width = max(max_width, mel_spec.shape[1])
            except Exception as e:
                print(f"Error loading {path}: {e}")

        self.max_width = max_width
        print(f"Maximum spectrogram width: {self.max_width}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mel_spec_path, label = self.samples[idx]

        # Load the mel spectrogram
        try:
            mel_spec = np.load(mel_spec_path)

            
            mel_spec = (mel_spec - np.mean(mel_spec)) / (np.std(mel_spec) + 1e-8)

            # Pad or resize to ensure uniform dimensions
            if self.max_width is not None:
                if mel_spec.shape[1] < self.max_width:
                    h
                    padded_spec = np.zeros((mel_spec.shape[0], self.max_width))
                    padded_spec[:, :mel_spec.shape[1]] = mel_spec
                    mel_spec = padded_spec
                elif mel_spec.shape[1] > self.max_width:
                    
                    mel_spec = mel_spec[:, :self.max_width]

           
            if self.transform:
                mel_spec = self.transform(mel_spec)
            else:
                # Add channel dimension for CNN (1 channel)
                mel_spec = np.expand_dims(mel_spec, axis=0)

            return torch.FloatTensor(mel_spec), torch.LongTensor([label])[0]

        except Exception as e:
            print(f"Error processing {mel_spec_path}: {e}")
            
            dummy = np.zeros((128, self.max_width or 500))
            dummy = np.expand_dims(dummy, axis=0)
            return torch.FloatTensor(dummy), torch.LongTensor([label])[0]


# Create weighted sampler to handle class imbalance
def create_weighted_sampler(dataset):
   
    labels = [label for _, label in dataset.samples]

    
    class_counts = np.bincount(labels)
    print(f"Class counts: {class_counts}")

    
    class_weights = 1. / class_counts
    weights = class_weights[labels]

    # Create sampler
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(weights),
        replacement=True
    )

    return sampler


#CNN-LSTM model with residual connections and batch normalization
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

        
        self.dropout_cnn = nn.Dropout(0.4)  
        self.dropout_lstm = nn.Dropout(0.4)  

        
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

        print(f"CNN output shape: {x.shape}")
        print(f"LSTM input size: {self.lstm_input_size}")

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
        x = x.permute(0, 2, 1, 3)  # [batch_size, height, channels, width]
        x = x.reshape(batch_size, time_steps, -1)  # [batch_size, time_steps, channels*width]

        # LSTM processing
        lstm_out, _ = self.lstm(x)  # [batch_size, time_steps, hidden_size*2]

        # Multi-head self-attention (reshape for compatibility)
        lstm_out = lstm_out.permute(1, 0, 2)  # [time_steps, batch_size, hidden_size*2]
        attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_output = attn_output.permute(1, 0, 2)  # [batch_size, time_steps, hidden_size*2]

        # Global average pooling over time dimension
        x = torch.mean(attn_output, dim=1)  # [batch_size, hidden_size*2]

        # Fully connected layers with batch normalization
        x = self.dropout_lstm(x)
        x = self.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_lstm(x)
        x = self.relu(self.bn_fc2(self.fc2(x)))
        x = self.fc3(x)

        return x



def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, save_path, patience=10):
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    # Early stopping variables
    early_stop_counter = 0

    # Initialize scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Mixed precision training
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                # Scale gradients and optimize
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation"):
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_val_loss = running_loss / total
        epoch_val_acc = correct / total
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)

        print(f"Epoch {epoch + 1}/{num_epochs} - "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

        # Update learning rate scheduler
        scheduler.step(epoch_val_acc)

        # Save the best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            early_stop_counter = 0

            # Save model checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': epoch_val_acc,
                'val_loss': epoch_val_loss,
                'train_acc': epoch_train_acc,
                'train_loss': epoch_train_loss
            }
            torch.save(checkpoint, os.path.join(save_path, 'best_model.pth'))
            print(f"Saved best model with validation accuracy: {best_val_acc:.4f}")
        else:
            early_stop_counter += 1

        # Early stopping check
        if early_stop_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    # Save the final model
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': epoch_val_acc,
        'val_loss': epoch_val_loss,
        'train_acc': epoch_train_acc,
        'train_loss': epoch_train_loss
    }
    torch.save(checkpoint, os.path.join(save_path, 'final_model.pth'))

    # Save the training history
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accs,
        'val_acc': val_accs
    }
    torch.save(history, os.path.join(save_path, 'training_history.pth'))

    return history


def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=[
        'Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad'
    ])

    return cm, report, all_preds, all_labels


def plot_training_history(history, save_path):
    plt.figure(figsize=(12, 5))

    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

   
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_history.png'))
    plt.close()


def plot_confusion_matrix(cm, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad'],
                yticklabels=['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
    plt.close()


def main():
    # Set paths
    mel_dir = 'cremad_mels'  # Path to the directory containing emotion folders
    save_dir = 'cremad_custom2'  # Directory to save model outputs

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Initialize dataset with directory scanning
    print("Loading mel spectrograms...")
    dataset = CREMADDataset(base_dir=mel_dir)

    # Get a sample to determine dimensions
    sample_data, _ = dataset[0]
    input_height, input_width = sample_data.shape[1], sample_data.shape[2]
    print(f"Sample mel spectrogram shape: {sample_data.shape}")

    # Split dataset into train, validation, and test
    train_indices, temp_indices = train_test_split(
        range(len(dataset)),
        test_size=0.3,
        random_state=42,
        stratify=[label for _, label in dataset.samples]
    )

    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=0.5,
        random_state=42,
        stratify=[dataset.samples[i][1] for i in temp_indices]
    )

    # Create subset datasets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    print(f"Training set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples")

    # Create weighted sampler for training set to handle class imbalance
    train_labels = [dataset.samples[i][1] for i in train_indices]
    class_counts = np.bincount(train_labels)
    class_weights = 1. / class_counts
    sample_weights = class_weights[train_labels]

    train_sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(train_indices),
        replacement=True
    )

    # Create dataloaders with increased batch size
    batch_size = 64  
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize the model 
    model = ImprovedCNNLSTMModel(
        num_classes=6,
        input_channels=1,
        input_height=input_height,
        input_width=input_width
    ).to(device)
    print(model)

    # Define loss function and optimizer with weight decay
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001,
                            weight_decay=1e-4) 

    # Train the model with early stopping
    print("Starting training...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=40,
        save_path=save_dir,
        patience=15  # Early stopping patience
    )

    # Plot training history
    plot_training_history(history, save_dir)

    # Load the best model for evaluation
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']} with validation accuracy: {checkpoint['val_acc']:.4f}")

    # Evaluate the model on the test set
    cm, report, all_preds, all_labels = evaluate_model(model, test_loader)

    # Plot confusion matrix
    plot_confusion_matrix(cm, save_dir)

    # Save classification report
    with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)

    print("Classification Report:")
    print(report)

    # Save the model in different formats
    torch.save(model.state_dict(), os.path.join(save_dir, 'model_state_dict.pth'))
    torch.save(model, os.path.join(save_dir, 'entire_model.pth'))

    print(f"All outputs saved to {save_dir}")


if __name__ == "__main__":
    main()
