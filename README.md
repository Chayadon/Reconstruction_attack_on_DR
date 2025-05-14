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
We will illustrate how to run our code on the following settings:
 - **Dataset** : the MNIST dataset
 - **Dataset size** : (Training, 4,000), (Testing, 500), (Validation, 500)
 - **Dimensionality reduction method** : Isomap
 - **Known-member set size(S)** : 9
1. Run `training_data.py` to generate two files: `X_recon.npy` and `y_recon.npy`. If you would like to modify the dataset, you can edit the code.
      ```bash
      python training_data.py
      ```
2. Run `main.py` to start the experiment.
      ```bash
      python main.py
      ```
3. If the code runs successfully, you should receive a result that resemble the following example.
      ```bash
      Training finished
      Mean Squared Error : 0.084983
      ```
> [!NOTE]
> Our provided code does not create any reconstructed images for comparison with the original data.
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
