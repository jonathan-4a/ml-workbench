import torch
from torch import nn, optim
from torchvision import models
from tqdm import tqdm
from torch.serialization import add_safe_globals


def train_model(
    dataloaders,
    arch="vgg16",
    learning_rate=0.001,
    hidden_units=512,
    epochs=5,
    gpu=False,
):
    """
    Trains a deep learning model and returns the trained model, optimizer, and criterion.

    Args:
        dataloaders (dict): Dictionary containing 'train' and 'valid' DataLoaders.
        arch (str): Model architecture ('vgg16' or 'densenet121').
        learning_rate (float): Learning rate for the optimizer.
        hidden_units (int): Number of hidden units in the custom classifier.
        epochs (int): Number of training epochs.
        gpu (bool): Whether to use GPU for training.

    Returns:
        model: The trained PyTorch model.
        optimizer: The optimizer used for training.
        criterion: The loss function used for training.
    """
    # Load pretrained model with updated weights
    if arch == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    elif arch == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # Define a new classifier
    input_size = (
        model.classifier[0].in_features
        if arch == "vgg16"
        else model.classifier.in_features
    )
    classifier = nn.Sequential(
        nn.Linear(input_size, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_units, 102),  # Output layer (102 flower classes)
        nn.LogSoftmax(dim=1),  # Use LogSoftmax for NLLLoss
    )
    model.classifier = classifier

    # Define loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # Move model to GPU if available
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)

    # Train the model
    for epoch in range(epochs):
        model.train()
        running_loss = 0

        # Wrap the training data loader with tqdm
        loop = tqdm(dataloaders["train"], desc=f"Epoch {epoch+1}/{epochs}", leave=True)

        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            logps = model(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Update the progress bar with the current loss
            loop.set_postfix(loss=loss.item())

        # Compute average training loss
        avg_train_loss = running_loss / len(dataloaders["train"])

        # Validation phase
        model.eval()
        validation_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in dataloaders["valid"]:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model(inputs)
                loss = criterion(logps, labels)
                validation_loss += loss.item()

                # Accuracy calculation
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                correct += equals.sum().item()
                total += labels.size(0)

        # Compute average validation loss and accuracy
        avg_valid_loss = validation_loss / len(dataloaders["valid"])
        valid_accuracy = correct / total * 100

        # Print training and validation statistics
        print(
            f"Epoch {epoch+1}/{epochs}.. "
            f"Train Loss: {avg_train_loss:.3f}.. "
            f"Valid Loss: {avg_valid_loss:.3f}.. "
            f"Valid Accuracy: {valid_accuracy:.2f}%"
        )

    return model, optimizer, criterion


# Example usage of saving the checkpoint
def save_checkpoint(model, filepath, arch, hidden_units, class_to_idx):
    """
    Saves the model checkpoint to a file.

    Args:
        model: The trained PyTorch model.
        filepath (str): Path to save the checkpoint file.
        arch (str): Model architecture.
        hidden_units (int): Number of hidden units in the custom classifier.
        class_to_idx (dict): Mapping of class labels to indices.
    """
    checkpoint = {
        "arch": arch,
        "hidden_units": hidden_units,
        "classifier": model.classifier,  # Save the custom classifier
        "state_dict": model.state_dict(),
        "class_to_idx": class_to_idx,
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, gpu=False):
    """
    Loads a checkpoint and rebuilds the model for inference.

    Args:
        filepath (str): Path to the checkpoint file.
        gpu (bool): Whether to use GPU for inference.

    Returns:
        model: The rebuilt PyTorch model ready for inference.
    """
    # Add necessary classes to the safe globals list
    add_safe_globals(
        [
            torch.nn.modules.container.Sequential,
            torch.nn.modules.linear.Linear,
            torch.nn.modules.dropout.Dropout,
            torch.nn.modules.activation.ReLU,
            torch.nn.modules.activation.LogSoftmax,
        ]
    )

    # Load the checkpoint with weights_only=True
    checkpoint = torch.load(filepath, weights_only=True)

    # Rebuild the model architecture
    arch = checkpoint["arch"]
    if arch == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    elif arch == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    # Replace the classifier with the saved custom classifier
    model.classifier = checkpoint["classifier"]

    # Load the model parameters
    model.load_state_dict(checkpoint["state_dict"])

    # Attach the class-to-index mapping
    model.class_to_idx = checkpoint["class_to_idx"]

    # Move model to GPU if requested
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)

    # Set the model to evaluation mode
    model.eval()

    return model
