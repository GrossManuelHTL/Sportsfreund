import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import argparse
import re
from sklearn.model_selection import train_test_split


class PoseClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        output, (hidden, _) = self.lstm(x)
        x = self.dropout(hidden[-1])
        return self.fc(x)


def load_and_preprocess_data(data_dir, exercise_name, max_sequence_length=120, test_size=0.2):
    """
    Load and preprocess keypoint data for a specific exercise.

    Args:
        data_dir: Directory containing keypoint data files
        exercise_name: Name of the exercise to train on
        max_sequence_length: Maximum sequence length for padding/truncation
        test_size: Proportion of data to use for testing

    Returns:
        X_train, X_test, y_train, y_test: Train/test split of data
    """
    X = []
    y = []
    sequence_lengths = []

    # Find all matching .npy files
    pattern = re.compile(f"{exercise_name}_(?:correct|incorrect)_\\d+\\.npy$")
    keypoint_files = [f for f in os.listdir(data_dir) if pattern.match(f)]

    if not keypoint_files:
        raise ValueError(f"No keypoint files found for exercise '{exercise_name}' in {data_dir}")

    print(f"Found {len(keypoint_files)} keypoint files for exercise '{exercise_name}'")

    # Load and preprocess each file
    for file in sorted(keypoint_files):
        file_path = os.path.join(data_dir, file)

        try:
            # Load keypoints
            keypoints = np.load(file_path)

            # Store sequence length for statistics
            sequence_lengths.append(keypoints.shape[0])

            # Truncate if necessary
            if keypoints.shape[0] > max_sequence_length:
                keypoints = keypoints[:max_sequence_length]

            # Pad if necessary
            if keypoints.shape[0] < max_sequence_length:
                pad_length = max_sequence_length - keypoints.shape[0]
                keypoints = np.pad(keypoints, ((0, pad_length), (0, 0)), mode='constant')

            # Add to dataset
            X.append(keypoints)

            # Determine label (0 for correct, 1 for incorrect)
            label = 0 if "correct" in file else 1
            y.append(label)

            print(f"Loaded {file}: {keypoints.shape} (Label: {'correct' if label == 0 else 'incorrect'})")

        except Exception as e:
            print(f"Error loading {file}: {e}")

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Report sequence length statistics
    if sequence_lengths:
        print(f"Sequence length statistics:")
        print(f"  Min: {min(sequence_lengths)}")
        print(f"  Max: {max(sequence_lengths)}")
        print(f"  Mean: {np.mean(sequence_lengths):.1f}")
        print(f"  Using max_sequence_length={max_sequence_length}")

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    print(f"Training set: {X_train.shape}, {y_train.shape}")
    print(f"Test set: {X_test.shape}, {y_test.shape}")

    return X_train, X_test, y_train, y_test


def train_model(data_dir, output_dir, exercise_name, epochs=50, batch_size=8,
                lr=0.001, hidden_size=128, num_layers=2, dropout=0.2, max_sequence_length=120):
    """Train a pose classifier for a specific exercise."""
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Load and preprocess data
    try:
        X_train, X_test, y_train, y_test = load_and_preprocess_data(
            data_dir, exercise_name, max_sequence_length
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Create model
    input_size = X_train.shape[2]  # Number of features per keypoint (usually 4: x, y, z, visibility)
    model = PoseClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )

    # Set up training
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Training loop
    best_accuracy = 0.0
    best_epoch = 0
    early_stop_patience = 10
    no_improve_count = 0

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss /= len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Evaluation
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        test_loss /= len(test_loader)
        test_accuracy = 100 * test_correct / test_total
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        # Update learning rate based on validation loss
        scheduler.step(test_loss)

        # Print metrics
        print(f"Epoch {epoch + 1}/{epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%")

        # Check for improvement
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch
            no_improve_count = 0

            # Save best model
            model_path = os.path.join(output_dir, f"{exercise_name}_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'input_size': input_size,
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'dropout': dropout,
                'max_sequence_length': max_sequence_length,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'best_accuracy': best_accuracy
            }, model_path)
            print(f"Model saved to {model_path}")
        else:
            no_improve_count += 1

        # Early stopping
        if no_improve_count >= early_stop_patience:
            print(f"Early stopping after {epoch + 1} epochs without improvement")
            break

    print(f"Training completed. Best accuracy: {best_accuracy:.2f}% at epoch {best_epoch + 1}")

    # Create a summary file
    summary_path = os.path.join(output_dir, f"{exercise_name}_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Exercise: {exercise_name}\n")
        f.write(f"Best accuracy: {best_accuracy:.2f}% at epoch {best_epoch + 1}\n")
        f.write(f"Input size: {input_size}\n")
        f.write(f"Hidden size: {hidden_size}\n")
        f.write(f"Num layers: {num_layers}\n")
        f.write(f"Dropout: {dropout}\n")
        f.write(f"Max sequence length: {max_sequence_length}\n")
        f.write(f"Training samples: {len(X_train)}\n")
        f.write(f"Test samples: {len(X_test)}\n")
        f.write(f"Class distribution (train): Correct={sum(y_train == 0)}, Incorrect={sum(y_train == 1)}\n")
        f.write(f"Class distribution (test): Correct={sum(y_test == 0)}, Incorrect={sum(y_test == 1)}\n")

    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a pose classifier model')
    parser.add_argument('--data', type=str, default='../data/poses',
                        help='Directory containing keypoint data files')
    parser.add_argument('--output', type=str, default='../data/models',
                        help='Output directory for trained models')
    parser.add_argument('--exercise', type=str, required=True,
                        help='Name of the exercise to train on')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--hidden-size', type=int, default=128,
                        help='Hidden size of LSTM')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--max-sequence-length', type=int, default=120,
                        help='Maximum sequence length for padding/truncation')

    args = parser.parse_args()

    train_model(
        data_dir=args.data,
        output_dir=args.output,
        exercise_name=args.exercise,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_sequence_length=args.max_sequence_length
    )