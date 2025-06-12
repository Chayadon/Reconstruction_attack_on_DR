# Reconstruction Neural Networks for Attacking Dimensionality Reduction Methods :space_invader:

 This repository contains the official code of the paper named:
 > Chayadon Lumbut and Donlapark Ponnoprat(2024). Investigating Privacy Leakage in Dimensionality Reduction Methods via Reconstruction Attack.

We provide the code for the neural network architecture and dataset generation, which you can further implement and extend for your own experiments.

## 🔧Installation
1. To run the provided code, the following main libraries need to be installed: `numpy`, `scikit-learn`, and `torch`.

## 📂Datasets
In this paper, we use the MNIST and NIH Chest X-ray datasets, which are publicly downloadable from the following:
1. [The MNIST dataset](https://keras.io/api/datasets/mnist/)
2. [The NIH Chest X-ray dataset](https://www.kaggle.com/datasets/nih-chest-xrays/data)

## 🤖Usage
We will illustrate how to run our code on the following settings:
 - **Dataset** : the MNIST dataset
 - **Dataset size** : {(Training, 4,000), (Testing, 500), (Validation, 500)}
 - **Dimensionality reduction method** : Isomap
 - **Known-member set size(S)** : 9
1. Run `main.py` to start the experiment. The program will automatically call `training_data.py` to generate two files: `X_recon.npy` and `y_recon.npy`. If you would like to modify the dataset or settings, you can edit the code in `training_data.py`.
      ```bash
      python main.py
      ```
2. If the code runs successfully, you should receive a result that resembles the following example.
      ```bash
      # The model has been trained through all epochs
      Training finished
      Mean Squared Error : 0.084983

      # Early stopping is activated
      Stop training at epoch 44!
      Mean Squared Error : 0.067679
      ```
> [!NOTE]
> Our provided code does not create any reconstructed images for comparison with the original data.
## 📚Citation
```bibtex
@article{LUMBUT2025104102,
title = {Investigating privacy leakage in dimensionality reduction methods via reconstruction attack},
journal = {Journal of Information Security and Applications},
volume = {92},
pages = {104102},
year = {2025},
issn = {2214-2126},
doi = {https://doi.org/10.1016/j.jisa.2025.104102},
url = {https://www.sciencedirect.com/science/article/pii/S2214212625001395},
author = {Chayadon Lumbut and Donlapark Ponnoprat},
keywords = {Dimensionality reduction, Manifold learning, Machine learning, Neural network, Reconstruction attack, Differential privacy},
abstract = {This study investigates privacy leakage in dimensionality reduction methods through a novel machine learning-based reconstruction attack. Employing an informed adversary threat model, we develop a neural network capable of reconstructing high-dimensional data from low-dimensional embeddings. We evaluate six popular dimensionality reduction techniques: principal component analysis (PCA), sparse random projection (SRP), multidimensional scaling (MDS), Isomap, t-distributed stochastic neighbor embedding (t-SNE), and uniform manifold approximation and projection (UMAP). Using both MNIST and NIH Chest X-ray datasets, we perform a qualitative analysis to identify key factors affecting reconstruction quality. Furthermore, we assess the effectiveness of an additive noise mechanism in mitigating these reconstruction attacks. Our experimental results on both datasets reveal that the attack is effective against deterministic methods (PCA and Isomap). but ineffective against methods that employ random initialization (SRP, MDS, t-SNE and UMAP). The experimental results also show that, for PCA and Isomap, our reconstruction network produces higher quality outputs compared to a previously proposed network. We also study the effect of additive noise mechanism to prevent the reconstruction attack. Our experiment shows that, when adding the images with large noises before performing PCA or Isomap, the attack produced severely distorted reconstructions. In contrast, for the other four methods, the reconstructions still show some recognizable features, though they bear little resemblance to the original images.}
}
```
