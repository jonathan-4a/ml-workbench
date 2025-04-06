import torch
from torchvision import datasets, transforms
import numpy as np
from PIL import Image

def load_data(data_dir):
    train_dir = data_dir + "/train"
    valid_dir = data_dir + "/valid"
    test_dir = data_dir + "/test"

    # Define transforms
    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Load datasets
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Define dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

    dataloaders = {"train": trainloader, "valid": validloader, "test": testloader}
    image_datasets = {"train": train_data, "valid": valid_data, "test": test_data}

    return dataloaders, image_datasets


def process_image(image_path):
    """
    Preprocess an image for inference.

    Args:
        image_path (str): Path to the input image.

    Returns:
        np.ndarray: Processed image as a NumPy array.
    """
    pil_image = Image.open(image_path)

    # Resize the image while maintaining aspect ratio
    width, height = pil_image.size
    aspect_ratio = width / height
    if width < height:
        new_width = 256
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = 256
        new_width = int(new_height * aspect_ratio)
    pil_image = pil_image.resize((new_width, new_height))

    # Crop the center of the image
    left = (new_width - 224) / 2
    top = (new_height - 224) / 2
    right = (new_width + 224) / 2
    bottom = (new_height + 224) / 2
    pil_image = pil_image.crop((left, top, right, bottom))

    # Convert to NumPy array and normalize
    np_image = np.array(pil_image) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    # Transpose dimensions to match PyTorch's expectations
    np_image = np_image.transpose((2, 0, 1))
    return np_image
