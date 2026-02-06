## Fingerprint/Measurements/Capacitive Embedding Extractor

## 💡 : Method
<!-- <div align="center">
<img width="400" alt="image" src="./vit.png">
</div> -->

The reconstructed HR fingerprints are used to train the ***Fingerprint Embedding Extractor***, which learns the mapping from the fingerprint image to embeddings such that the fingerprint embeddings of the same finger are pulled together, while the fingerprint embeddings of distinct fingers are pushed away.

The SPI measurements are essentially a time-series signal that contains the light intensity measured for the sequence of the illumination patterns. We thus develop our ***Measurements Embedding Extractor*** based on a 1D-CNN since the 1D-CNN can capture the temporal fluctuations of light as it interacts with the finger.

We develop our ***Capacitive Embedding Extractor*** based on an MLP, which treats the entire capacitive image as a unique global signature rather than trying to find local edges that do not exist at such resolution.

## ⚙ : Setup
First create a new conda environment

    conda env create -f environment.yml
    conda activate embedding

Install PyTorch (Linux OS as an example)

    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

## ☕️ : Training and Evaluation
You can train and evaluate the model by:

    python src/train.py
