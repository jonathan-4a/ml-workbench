import argparse
from utils import load_data
from model_utils import train_model, save_checkpoint


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train a deep learning model on a dataset."
    )
    parser.add_argument("data_dir", type=str, help="Path to the data directory")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="vgg16",
        choices=["vgg16", "densenet121"],
        help="Model architecture",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--hidden_units",
        type=int,
        default=512,
        help="Number of hidden units in the classifier",
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")

    args = parser.parse_args()

    # Load data
    print("Loading data...")
    dataloaders, image_datasets = load_data(args.data_dir)

    # Train the model
    print("Training the model...")
    model, optimizer, criterion = train_model(
        dataloaders,
        arch=args.arch,
        learning_rate=args.learning_rate,
        hidden_units=args.hidden_units,
        epochs=args.epochs,
        gpu=args.gpu,
    )

    # Save the checkpoint
    print("Saving the checkpoint...")
    save_checkpoint(
        model=model,
        filepath=f"{args.save_dir}/checkpoint.pth",  # Save the checkpoint file in the specified directory
        arch=args.arch,  # Model architecture (e.g., "vgg16")
        hidden_units=args.hidden_units,  # Number of hidden units in the classifier
        class_to_idx=image_datasets["train"].class_to_idx,  # Class-to-index mapping
    )
    print(f"Checkpoint saved to {args.save_dir}/checkpoint.pth")


if __name__ == "__main__":
    main()
