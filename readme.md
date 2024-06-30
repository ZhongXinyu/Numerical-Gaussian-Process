# Gaussian Process Project

## Introduction
This is the repo for the Gaussian Process Project

## Getting Started

### Installation of environment
1. install environment from environment.yml
```bash
conda env create -f environment.yml
```

2. activate the environment
```bash
conda activate GaussianProcessProject
```

## Folder Documentation

### `testdata` Directory

**Purpose:**
Contains example test data for equations.

**Contents:**
- `burger_eqn.json`: Example data related to the Burger Equation.
- `wave_eqn.json`: Example data related to the Wave Equation.

### `report` Directory

**Purpose:**
Contains reports and analysis notebooks.

#### `solver` Subdirectory

**Contents:**
- `burger_equation_solve_arcSin.ipynb`: Notebook for solving the Burger Equation using the arcSin method.
- `burger_equation_solve_RBF.ipynb`: Notebook for solving the Burger Equation using Radial Basis Functions (RBF).
- `wave_equation_solve_RBF.ipynb`: Notebook for solving the Wave Equation using Radial Basis Functions (RBF).
- `kernels.py`: Contains the kernel functions used in the notebooks.

#### `data` Subdirectory

**Contents:**
- `burger_equation.ipynb`: Notebook for analysis or visualization related to the Burger Equation.
- `wave_equation.ipynb`: Notebook for analysis or visualization related to the Wave Equation.
- `ODE.py`: Contains the ODE class for the Burger Equation and Wave Equation.

#### `plot_graphs.ipynb`

**Purpose:**
Notebook for plotting graphs and visualizations of results


## Usage
To run the project, you can use the example notebooks in the `src\solver` directory.

A new function is written under the GPJax framework to update the variance of the artificial data points. The new function is called `predict_with_prev_cov()`, and can be found within `src/utilities.py`.

To be able to use the new function, the function `predict_with_prev_cov()` should be pasted under the `ConjugatePosterior()` class in file `gpjax/gps.py`. It should be placed after the `predict()` function. It is important to note that the new function is not a part of the GPJax library, and it is only used in this project and should be copied manually.

The notebooks contain the code for the Gaussian Process to solve linear partial differential equations, such as the Burger Equation and the Wave Equation.
- Before running the notebooks, make sure to install the required packages in the environment.yml file, and edit the parameters in the notebooks as needed.

The `plot_graphs.ipynb` notebook allows you to plot the results of the numerical Gaussian Process against the actual solution, or against different parameters.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details


