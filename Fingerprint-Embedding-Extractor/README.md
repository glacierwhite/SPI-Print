## Fingerprint Embedding Extractor

## 💡 : Method
<!-- <div align="center">
<img width="400" alt="image" src="./vit.png">
</div> -->

The reconstructed high-resolution fingerprints are used to train the ***Fingerprint Embedding Extractor***, which learns the mapping from the fingerprint image to embeddings such that the fingerprint embeddings of the same finger are pulled together, while the fingerprint embeddings of distinct fingers are pushed away.

## ⚙ : Setup
First create a new conda environment

    conda env create -f environment.yml
    conda activate vit

Install PyTorch (Linux OS as an example)

    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

## ☕️ : Training
You can train the model by:

    python src/train.py

## 💻 : Test
You can test the model by:

    python src/test.py --load_model ./checkpoints/model_best.pth
