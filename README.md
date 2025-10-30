Lecture 1 - CNNs [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/HebaNAS/D2xDL-Week8/HEAD?urlpath=%2Fdoc%2Ftree%2Fcnn.ipynb)

Project examples:
1. Regression, simple ML algos: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/HebaNAS/D2xDL-Week8/HEAD?urlpath=%2Fdoc%2Ftree%2Fml_project_example_regression.ipynb)
2. Classification, deep NN (CNN): [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/HebaNAS/D2xDL-Week8/HEAD?urlpath=%2Fdoc%2Ftree%2Fml_project_example_image_classification.ipynb)

To run these notebooks locally, you will need to setup the python enivronment with the correct dependencies. These are already included in this repo.

## Running locally

1. Download the repository

2. Install UV package manager from: [https://docs.astral.sh/uv/#highlights](https://docs.astral.sh/uv/#highlights)

3. Within the repository run the following command:
```
uv venv --python 3.12
```
or just

```
uv venv
```

4. Run
```
uv sync
```

5. Run
```
uv run jupyter notebook
```
