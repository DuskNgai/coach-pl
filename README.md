# Coach-pl

Coach-pl is a [PyTorch-Lightning](https://lightning.ai/docs/pytorch/stable/) based deep learning framework for training and evaluating models.
It is designed to be flexible, allowing for quick prototyping and experimentation.

If you have any questions or suggestions, please feel free to contact me through github issues.

## Requirements

These are the dependencies for the package:

```bash
conda install lightning rich
pip install deepspeed fvcore omegaconf
```

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
3. import it in the `__init__.py` file in the corresponding parent directory.
```python
# In `model/example.py`
from coach_pl.configuration import configurable
from coach_pl.models import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class ExampleModel(nn.Module):
    @configurable
    def __init__(self, arg) -> None:
        super().__init__()

    @classmethod
    def from_config(cls, cfg) -> dict[str, Any]:
        return {"arg": cfg.arg}

# In `model/__init__.py`
from .example import ExampleModel
```

We also provide a `yapf` configuration file `.style.yapf` for code formatting.

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
