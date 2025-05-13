# Reconstruction Neural Network for Attacking Dimensionality Reduction Methods

 This repository contains the official code of the paper named:
 > Investigating Privacy Leakage in Dimensionality Reduction Methods via Reconstruction Attack.

This study investigates privacy leakage in dimensionality reduction methods through a novel machine learning-based reconstruction attack. Employing an \emph{informed adversary} threat model, we develop a neural network capable of reconstructing high-dimensional data from low-dimensional embeddings.
We evaluate six popular dimensionality reduction techniques: PCA, sparse random projection (SRP), multidimensional scaling (MDS), Isomap, -SNE, and UMAP. Using both MNIST and NIH Chest X-ray datasets, we perform a qualitative analysis to identify key factors affecting reconstruction quality. Furthermore, we assess the effectiveness of an additive noise mechanism in mitigating these reconstruction attacks.




## Citation
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
