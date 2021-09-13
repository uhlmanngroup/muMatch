# µMatch: 3D shape correspondence for microscopy data

 µMatch (microMatch) is a shape correspondense library for a variety of mesh types.

## Installation

 - Clone the repository.
 - Create a conda virtual environment and install the package
 - Install the project using the source
```bash
git clone <this repo url>
cd muMatch
conda env create -f environment.yml
conda activate mumatch
pip install .
```

## Installation for development mode

 - Clone the repository.
 - Create a conda virtual environment and install the package
 - Install the project using the source in [editable mode](https://packaging.python.org/guides/distributing-packages-using-setuptools/#working-in-development-mode)
```bash
git clone <this repo url>
cd muMatch
conda env create -f environment.yml
conda activate mumatch
pip install --editable .
```

### Optional: automatic house-keeping

To maintain a good code base, you can setup [pre-commit hooks](https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks) to:
 - format your code (using a code formatter like black)
 - inspect your code for potential caveats (using a linter like flake8)
 - canonically sort your import (using isort)

To setup pre-commit setup, you can type when your environment is activated:
```bash
pip install pre-commit black flake8 isort
pre-commit install
```


### How to cite

If you use µMatch, please cite us as follows:

> J. Klatzow, G. Dalmasso, N. Martinez-Abadias, J. Sharpe, and V. Uhlmann, "μMatch: 3D shape correspondence for microscopy data", 2021.
