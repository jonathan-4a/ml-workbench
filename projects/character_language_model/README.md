# Character-level autoregressive model

This project estimates the next character from the current character using a smoothed bigram probability table. It downloads the small `stas/gutenberg-100` dataset from Hugging Face, then keeps only two public-domain books: *Frankenstein* and *The Time Machine*.

The notebook covers corpus inspection and cleaning, PyTorch character encoding and tensors, an ordered train/validation split, unigram and count-based bigram baselines, a PyTorch bigram model trained with cross-entropy and gradient descent, validation negative log-likelihood and perplexity, a compact probability heatmap, and temperature-controlled generation. NumPy is not used for the model or training path; the only conversion is to hand a final tensor to Matplotlib for plotting.

Run `character_level_bigram.ipynb` from top to bottom. The first data cell downloads the selected records through the Hugging Face `datasets` library. The hosted dataset is described at https://huggingface.co/datasets/stas/gutenberg-100.
