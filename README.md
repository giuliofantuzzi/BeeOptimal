<div align="center">
  <img src="assets/logo_BeeOptimal.png" alt="Logo" width="250">
  <h3 align="center">BeeOptimal</h3>
  <p align="center">
    A Python implementation of the Artificial Bee Colony (ABC) optimization algorithm
    <br />
      <a href="LINK TO INSERT"><strong>Explore the docs »</strong></a>
    <br />
  </p>
</div>

# About

**BeeOptimal** is an open-source Python package that implements the *Artificial Bee Colony (ABC)* algorithm, a population-based optimization method 
inspired by the foraging behavior of honeybee swarms and designed to solve complex optimization problems efficiently. Whether you are tackling high-dimensional 
search spaces, multi-modal objective functions, or simply need a reliable optimizer, **BeeOptimal** offers a user-friendly and customizable solution for your needs.

# Installation

Before installing the package, make sure you have **Python 3.12 or higher** installed on your system. 
In case you want to avoid any conflicts with your system's Python packages, you might want to create (and activate) a dedicated virtual environment:

```bash
python -m venv /path/to/virtual/environment/beeoptimal_env
source /path/to/virtual/environment/beeoptimal_env/bin/activate
```


## Installing via PIP

You can install the package from the *Python Package Index (PyPI)* via pip:

```bash
pip install beeoptimal
```

## Installing from source

1. Clone the repository:
   
   ```bash
   git clone https://github.com/giuliofantuzzi/BeeOptimal.git
   ```

2. Move into the repository directory and install the package with:
   
   ```bash
   cd BeeOptimal/
   pip install .
   ```

## Optional Dependencies

In addition to the core functionalities, this package offers optional dependencies for specific use cases.

To build and work with the documentation, you can install the package with the docs extra:

```bash
pip install beeoptimal[docs]
```
To use the tutorials and their required dependencies, install the package with the tutorials extra:
  
```bash
pip install beeoptimal[tutorials]
```

To install both the documentation and the tutorials, you can use directly:

```bash
pip install beeoptimal[docs,tutorials]
```

> [!NOTE]
> The same syntax can be followed when installing from source. Moreover, if you're using the `zsh` shell, you will need to wrap the extras in quotes to prevent conflicts with shell globbing (unquoted square brackets ([ ]) are used for pattern matching in `zsh`).