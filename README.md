# Official source code for our paper "Topological Neural Discrete Representation Learning à la Kohonen"

## TLDR

If you want to reuse the KSOM layer, look at `layers/som_vector_quantizer.py`. It has no external dependencies and supports multi-GPU training.

An example:

```python
from layers.som_vector_quantizer import SOMGeometry, Grid, HardSOM, HardNeighborhood

geometry = SOMGeometry(
    Grid(2),
    HardNeighborhood(0.1)
)

quantizer = HardSOM(128, 512, 0.99, geometry)

loss, output, perplexity, _ = qunatizer(input)
```

## Installation

This project requires Python 3 and PyTorch 1.8.

```bash
pip3 install -r requirements.txt
```

Create a Weights and Biases account and run
```bash
wandb login
```

More information on setting up Weights and Biases can be found on
https://docs.wandb.com/quickstart.

For plotting, LaTeX is required (to avoid Type 3 fonts and to render symbols). Installation is OS specific.

## Usage

The code makes use of Weights and Biases for experiment tracking. In the "sweeps" directory, we provide sweep configurations for all experiments we have performed. The sweeps are officially meant for hyperparameter optimization, but we use them to run 10 instances of each experiment.

To reproduce our results, start a sweep for each of the YAML files in the "sweeps" directory. Run wandb agent for each of them in the main directory. This will run all the experiments, and they will be displayed on the W&B dashboard.

### Re-creating plots from the paper

Edit config file "paper/config.json". Enter your project name in the field "wandb_project" (e.g. "username/modules").

Run the script of interest within the "paper" directory. For example:

```bash
cd paper/kohonen
python3 compare_init.py
```

