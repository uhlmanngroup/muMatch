# Mesh correspondence library

Library to include correspondence techniques for a variety of mesh types.

## Installation step

 - Clone the repository.
 - Create a conda virtual environment and install the package
 - Install the project using the source
```bash
git clone <this repo url>
cd mesh_correspondence_library
conda env create -f environment.yml
conda activate meshcorr
pip install .
```

## Installation step for development mode

 - Clone the repository.
 - Create a conda virtual environment and install the package
 - Install the project using the source in [editable mode](https://packaging.python.org/guides/distributing-packages-using-setuptools/#working-in-development-mode)
```bash
git clone <this repo url>
cd mesh_correspondence_library
conda env create -f environment.yml
conda activate meshcorr
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


---

## TODO-list

### Geodesic Matching

1. Geodesic optimiser - start at low time resolution.


### Improvements

1. Collective optimisation - update
3. Re-add curvature into optimisation??
4. Compare Deep Learning vs C optimisation.
5. Implement SHOT decriptor.

### General

1. Benchmark on Teeth, TOSCA and FAUST
