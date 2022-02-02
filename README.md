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

## Getting started

We provide an example analysis pipeline that can be run by calling

```bash
python run_pipeline.py
```

This file will run all steps of the microMatch pipeline on the example data we provide in the `example_data` folder and can be used as a basis to use microMatch on your own data. It also illustrates some basic shape analysis outputs that can be retreived once correspondence has been established.

We also provide a further example of visualization of the correspondence map in the jupyter notebook `scalar_plotting.ipynb`. Once `run_pipeline.py` has been run once, `scalar_plotting.ipynb` can be explored in your browser through the jupyter notebook interface by running

```bash
jupyter notebook
```

with the muMatch conda environment activated (see Installation above).


## Troubleshooting

- If you adapt `run_pipeline.py` directly to work on your own files, make sure that your file names do not include full stops, dashes or underscores (the shorter the name, the less risk of problems). 
- It is possible to transfer data from Imaris (.wrl) to meshlab (conversion from .wrl to .stl) and use .stl in muMatch.


## Optional: automatic house-keeping

To maintain a good code base, you can setup [pre-commit hooks](https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks) to:
 - format your code (using a code formatter like black)
 - inspect your code for potential caveats (using a linter like flake8)
 - canonically sort your import (using isort)

To setup pre-commit setup, you can type when your environment is activated:
```bash
pip install pre-commit black flake8 isort
pre-commit install
```

## Running the pipeline on example data

With the mumatch environment activated: 
```bash
python run_pipeline.py
```

## How to cite

If you use µMatch, please cite us as follows:

> J. Klatzow, G. Dalmasso, N. Martinez-Abadias, J. Sharpe, and V. Uhlmann, "μMatch: 3D shape correspondence for microscopy data", Frontiers in Computer Science, 2021. https://doi.org/10.3389/fcomp.2022.777615
