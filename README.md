# Regularization and Stability

Code for the report on ridge regression, stability and generalization (Bousquet & Elisseeff, 2002).
The report is in `report/`.

## Setup

Tested with Python 3.13.

    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

## Reproduce the results

Run from the root of the repository, in this order:

    python src/run_experiment.py            # sweep over alpha
    python src/cross_validation.py          # selection of alpha
    python src/plot_results.py              # main figures
    python src/extensions/effect_of_n.py    # extension on the training set size

Tables and figures are saved in `results/`. The modules `src/datasets.py`, `src/models.py` and
`src/stability.py` are imported by the scripts above and do not need to be run. All the random choices use fixed seeds, so the
numbers of the report are reproduced exactly.
The figures are saved as PNG files in `results/` (no window is shown).
