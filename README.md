# Matrix Quantum Mechanics with Qutip

The quantum mechanics of matrix models is solved using the Quantum Toolbox in Python ([qutip](www.qutip.org)).
Results are reported in the publication [Rinaldi et al. (2021)](www.arxiv.org/abs/2108.02942).
Consider the citation in [Cite](#cite).

We select two matrix models to study:

- a bosonic 2-matrix model with gauge group SU(2) 
- a supersymmetric 2-matrix model with gauge group SU(2)

Models with larger SU(N) groups can also be studied, but they require larger computational resources.

The Hilbert space of the bosonic 2-matrix model with SU(2) gauge group truncated to cutoff Λ (maximum number of modes for each boson) has dimension Λ⁶ because there are 6 bosonic degrees of freedom.
With a regular desktop (64GB of RAM) it is possible to reach cutoffs Λ=14 or slightly larger.

# Code

## Installation

There is a `environment.yml` file that can be used to build a `conda` python environment with all the dependencies needed to run the scripts and notebooks in this repository.

If you have `conda` installed (see [this link](https://docs.conda.io/projects/conda/en/latest/) if you need help), you can run 
```bash
conda env create -f environment.yml
```
and then check that the environement has been created
```bash
conda env list
```

If the environment appears in the output of the command above, you can activate it with
```bash
conda activate qutip-env
```
and then run the scripts and notebooks in this repository.

## Scripts

To generate the spectrum of the bosonic matrix model with a specific list of cutoffs and for a specific list of 't Hooft coupling constants you can run the [scripts/bmn2_bos_su2.py](./scripts/bmn2_bos_su2.py) script from the command line

For a list of options to pass on the command line, run
```bash
python scripts/bmn2_bos_su2.py --help
```

An example, you can run
```bash
python scripts/bmn2_bos_su2.py --num_eigs=20 --L_range=[3,5,7] --l_range=[0.2] --penalty=False
```
to get the first 20 eigenstates of the truncated Hamiltonian with cutoff Λ equal to 3, 5, and 7 at coupling constant 0.2 and without introducing a penalty term for the gauge non-singlet states.

Similarly, you can study the spectrum of the supersymmetric matrix model with the script [scripts/bmn2_su2.py](./scripts/bmn2_su2.py), which takes a similar list of arguments from the command line.

The rest of the scripts are just utilities for making plots.

**Note**: you can take advantage of multi-threading on a single node by changing the `OMP_NUM_THREAD` environment variable before calling `python`.
For example:
```bash
export OMP_NUM_THREADS=12 ; python scripts/bmn2_bos_su2.py --num_eigs=20 --L_range=[3,5,7] --l_range=[0.2] --penalty=False
```
will run the script using 12 cores.
## Notebooks

The Jupyter notebooks [QUTIP_bosonic_matrices](./notebooks/QUTIP_bosonic_matrices.ipynb) and [QUTIP_susy_matrices](./notebooks/QUTIP_susy_matrices.ipynb) give an interactive explanation on how to build the regularized Hamiltonians of the matrix models using `qutip`.

If you do not want to build the python environment to run them (see [Installation](#installation)) you can use [Google Colaboratory](colab.research.google.com) to run them in your web browser.
Use the two links below to open the notebooks in a new tab:

- Bosonic model: [![Bosonic model in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1U_H6Fl9AWkUsMLZQ_bawVvRT-vJDTHz5?usp=sharing)

- Supersymmetric model: [![Susy model in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1TXSzdcUGudCVJAQZTle1cPyhSN-BZQAr?usp=sharing)


# Cite

If you use this code (or parts of it), please consider citing our paper:
```bibtex
@misc{rinaldi2021matrixmodels,
    title   = {Matrix Model simulations using Quantum Computing, Deep Learning, and Lattice Monte Carlo}, 
    author  = {Rinaldi, Enrico and Han, Xizhi and Hassan, Mohammad and Feng, Yuan and Nori, Franco and McGuigan, Michael and Hanada, Masanori},
    year    = {2021},
    eprint  = {2108.02942},
    archivePrefix = {arXiv},
    primaryClass = {quant-ph}
}
```