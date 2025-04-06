import argparse
import torch
from utils import process_image
from model_utils import load_checkpoint


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Predict the class of an image using a trained model."
    )

    # Required arguments
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("checkpoint", type=str, help="Path to the model checkpoint")

    # Optional arguments
    parser.add_argument(
        "--top_k", type=int, default=5, help="Return top K most likely classes"
    )
    parser.add_argument(
        "--category_names",
        type=str,
        default="cat_to_name.json",
        help="Path to a JSON file mapping categories to real names",
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference")

    args = parser.parse_args()

    # Load the model checkpoint
    print("Loading the model checkpoint...")
    model = load_checkpoint(args.checkpoint, gpu=args.gpu)

    # Preprocess the input image
    print("Processing the input image...")
    processed_image = process_image(args.image_path)

    # Predict the top-K classes
    print("Making predictions...")
    probs, classes = predict(processed_image, model, topk=args.top_k, gpu=args.gpu)

    # Map class indices to real names if provided
    class_names = None
    if args.category_names:
        import json

        with open(args.category_names, "r") as f:
            cat_to_name = json.load(f)
        class_names = [cat_to_name[cls] for cls in classes]

    # Print the results
    print("\nTop predictions:")
    for i, (prob, name) in enumerate(zip(probs, class_names or classes), start=1):
        print(f"{i}. {name} ({prob:.3f})")


def predict(image, model, topk=5, gpu=False):
    """
    Predict the class of an image using a trained model.

    Args:
        image (numpy.ndarray): Processed image as a NumPy array.
        model: The trained PyTorch model.
        topk (int): Number of top predictions to return.
        gpu (bool): Whether to use GPU for inference.

    Returns:
        probs (list): Probabilities of the top-K classes.
        classes (list): Class indices of the top-K predictions.
    """
    # Move model and data to GPU if available
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Convert image to tensor and add batch dimension
    image = torch.from_numpy(image).unsqueeze(0).float().to(device)

    # Perform inference
    with torch.no_grad():
        output = model(image)

        # Apply softmax to convert logits to probabilities
        ps = torch.softmax(output, dim=1)

        # Get the top-K probabilities and class indices
        top_p, top_class = ps.topk(topk, dim=1)

    # Extract probabilities and class indices
    probs = top_p.cpu().numpy()[0]
    classes = top_class.cpu().numpy()[0]

    # Map class indices to original dataset indices
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    classes = [idx_to_class[cls] for cls in classes]

    return probs, classes


if __name__ == "__main__":
    main()
