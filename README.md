# Reconstruction Neural Network for Attacking Dimensionality Reduction Methods

 This repository contains the official code of the paper named:
 > Chayadon Lumbut and Donlapark Ponnoprat(2024). Investigating Privacy Leakage in Dimensionality Reduction Methods via Reconstruction Attack.

We provide the code for the neural network architecture and dataset generation, which you can further implement and extend for your own experiments.

## 🔧Installation
1. To run the provided code, the following main libraries need to be installed: `numpy`, `scikit-learn`, and `torch`

## 📂Datasets
In this paper, we use the MNIST and NIH Chest X-ray datasets, which are publicly downloadable from the following:
1. [The MNIST dataset](https://keras.io/api/datasets/mnist/)
2. [The NIH Chest X-ray dataset](https://www.kaggle.com/datasets/nih-chest-xrays/data)

## 🤖Usage
We will illustrate how to run our code on these settings:
 - Dataset : the MNIST dataset
 - Dimensionality reduction method : Isomap
 - Known-member set size : 9
1. Run `training_data.py` to generate two files called `X_recon.npy` and `y_recon.npy`. If you want to change the dataset, you can edit the code.
```bash
python training_data.py
```
2. Run `main.py` to 

## 📚Citation
```bibtex
@misc{lumbut2024investigatingprivacyleakagedimensionality,
      title={Investigating Privacy Leakage in Dimensionality Reduction Methods via Reconstruction Attack}, 
      author={Chayadon Lumbut and Donlapark Ponnoprat},
      year={2024},
      eprint={2408.17151},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2408.17151}, 
}
```
