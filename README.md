# Coach-pl

Coach-pl is a PyTorch-Lightning based deep learning framework for training and evaluating models.
It is designed to be easy to use and flexible, allowing for quick prototyping and experimentation.

Note that the license of this project is **GPL-3.0**.

## Setup

To install the package, run the following command:

```bash
conda env create -f environment.yaml
conda activate coach-pl
```

## Usage

This package is designed to be a package that you can import and use in your own projects.
Currently, the package is supported as git submodule:

```bash
git submodule add ...
```

To use this package, one should create at least:
1. a pytorch style dataset
2. a pytorch-lightning style module
3. a yaml configuration file

