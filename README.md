# Coach-pl

Coach-pl is a PyTorch-Lightning based deep learning framework for training and evaluating models.
It is designed to be flexible, allowing for quick prototyping and experimentation.

If you have any questions or suggestions, please feel free to contact me through github issues.

## Usage

Currently, the package is supported as git submodule:

```bash
git submodule add https://github.com/DuskNgai/coach-pl.git coach_pl
```

To use this package, one should create at least:
1. a pytorch style dataset.
2. a pytorch style model.
3. a pytorch-lightning style module.
4. a yaml configuration file.

For each dataset, model, and module class, one should make it
1. registered in the corresponding registry.
2. configurable with the `@configurable` decorator and `from_config` method.
```python
from coach_pl.configuration import configurable
from coach_pl.models import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class ExampleModel(nn.Module):
    @configurable
    def __init__(self, arg):
        super().__init__()

    @classmethod
    def from_config(cls, cfg):
        return {"arg": cfg.arg}
```

## Citation

If you find this package useful, please consider citing it:

```bibtex
@misc{coach-pl,
  author = {Dusk Ngai},
  title = {Coach-pl: A PyTorch-Lightning Based Deep Learning Framework},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
}
```

Note that the license of this project is **GPL-2.0**.
