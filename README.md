# Applying recent machine learning approaches toaccelerate the Algebraic Multigrid method forfluid simulations

This repository contains the scripts and code necessary to reproduce the dataset and findings in our paper,

```
Louw, T.R. and McIntosh-Smith, S.N. Applying recent machine learning approaches to accelerate the Algebraic Multigrid method for fluid simulations, Smoky Mountains Computational Sciences & Engineering Conference (SMC2021), 2021
```

## Prior work
Our paper relies on implementations described in two papers:

```
Greenfeld, D., Galun, M., Kimmel, R., Yavneh, I., Basri, R.: Learning to Optimize Multigrid PDE Solvers.  36th ICML 2019 June, 4305–4316 (2019)
```
with its associated GitHub repository at https://github.com/danielgreenfeld3/Learning-to-optimize-multigrid-solvers

and

```
Luz, I., Galun, M., Maron, H., Basri, R., Yavneh, I.: Learning Algebraic Multigrid Using Graph Neural Networks. PMLR pp. 6489–6499 (2020)
```
with its associated repository at https://github.com/ilayluz/learning-amg

## Dataset
In our work, we use the 10,000 items in the [ThingI10K dataset](https://arxiv.org/abs/1605.04797), which have been processed into valid tetragonal meshes by the [fTetWild](https://github.com/wildmeshing/fTetWild) tool. The input dataset of fTetWild-processed meshes is available from the fTetWild Github page.
Last known correct download link: https://drive.google.com/file/d/13zmGxikHiiSv9-eu8wZDTOWtPmR-KV5b/view?usp=sharing

*NOTE: This input dataset is provided by the fTetWild authors, and we claim no ownership of it*

In our work, we use FeniCS to run a simple 3D fluid simulation on each geometry in the dataset, and 
capture the large, sparse matrix systems which result from the velocity and pressure correction calcluations. We are interested in solving these using AMG.

Running these simple simulations produces a dataset of 30,000 large sparse matrices, which we capture and store in [compressed Numpy format](https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html) (.npz). Even with just one set of simulation parameters (choice of Reynolds number etc.), this resulting dataset is approx 188GiB in size. 

This repository contains the scripts for you to reproduce this dataset. These are contained in the [dataset](./dataset) directory.

## Modifications to the networks in Greenfeld, et al.
In our paper, we describe additional network architectures for use with the approach from Greenfeld et al. 

Code for these is in the [greenfeld-modifications](./greenfeld-modifications) directory

## Modifications to the sources in Luz, et al.
We use TF2 and a more recent versions of the Deepmind Graphnets library in our work than the authors did. We describe these changes in the [luz-modifications](./luz-modifications) directory

## Experiments and analysis
Notebooks for our analysis are in the [experiments](./experiments) directory.
Our work was done in compute instances on the Google Cloud Platform, and uses storage in cloud buckets. You will need appropriate access to GCP if you want to use our scripts exactly.