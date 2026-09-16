# epymodelingsuite

Models and calibrators for forecasting and scenario modeling. A codebase for large workloads with epydemix and expanded functionality.

## Installation

To install directly from GitHub (recommended for normal usage), run `pip install git+ssh://git@github.com/mobs-lab/epymodelingsuite.git`. You must have an SSH keypair for your machine enrolled with GitHub and have access to the [repo](https://github.com/mobs-lab/epymodelingsuite).

## Development

### Environment Setup

**Option 1: uv**
```bash
make install-dev    # Creates isolated .venv in project directory
uv run pytest       # Run commands via uv
```
uv manages its own `.venv`. There is no need to activate anything.

**Option 2: pip/conda**
```bash
conda activate myenv    # Activate your environment first
make install-dev-pip    # Installs locked dependencies into active env
pytest                  # Run commands directly
```
Installs into whatever environment is currently active (conda, venv, system Python).

### Lock Files

- `uv.lock` is the single source of truth; `requirements.lock.txt` is exported from it.
- Run `make lock` to update after changing `pyproject.toml`.

### Jupyter Kernels

If you installed with uv (option 1) and want to use Jupyter notebooks:
- **Using notebooks in VSCode**: VSCode auto-detects the `.venv` directory. Select it as the Python interpreter/kernel.
- **Running JupyterLab from uv**: Start new JupyterLab with `uv run --with jupyterlab jupyter lab`. The kernel is available automatically.
- **Using an existing JupyterLab installation**: If you have JupyterLab set up separately, register the kernel manually:
  ```bash
  uv run python -m ipykernel install --user --name epymodelingsuite --display-name "epymodelingsuite"
  ```

## Documentation


