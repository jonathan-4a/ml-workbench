# Flower image classifier

This project is the applied image-classification project from Udacity's AI Programming with Python course. It classifies images from the Oxford 102 Flowers dataset using transfer learning.

The convolutional feature extractor comes from a pretrained VGG16 or DenseNet121 model. Its parameters are frozen, and a new feed-forward classifier is trained for the 102 flower categories. The project includes both the training path and a command-line inference path.

The notebook contains the original end-to-end experiment. The Python modules separate the same work into reusable data loading, model construction, checkpointing, training, and prediction code.

The dataset and trained checkpoints are intentionally not stored here. Download the Oxford 102 Flowers archive manually from [Udacity's dataset mirror](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz), then extract it in this directory:

```bash
curl -L https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz \
  | tar -xzf -
```

The command is documented for convenience only; nothing in the project downloads data automatically. After extraction, pass the `flower_data` directory to `train.py`.

## Files

- `flower_classifier.ipynb` — notebook version of the project
- `train.py` — command-line training entry point
- `predict.py` — command-line top-k prediction
- `model_utils.py` — model construction, training, and checkpoint loading
- `utils.py` — data transforms and image preprocessing
- `cat_to_name.json` — category id to flower-name mapping

## Example commands

```bash
python train.py /path/to/flower_data --arch vgg16 --epochs 5 --save_dir checkpoints
python predict.py image.jpg checkpoints/checkpoint.pth --category_names cat_to_name.json --top_k 5
```
