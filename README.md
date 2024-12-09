<p align="center">
  <a href="https://github.com/giuliofantuzzi/BeeOptimal">
    <!-- <img src="https://github.com/giuliofantuzzi/BeeOptimal/raw/main/assets/logo_BeeOptimal.png" alt="Logo" width="250" height="250"> -->
    <img src="assets/logo_BeeOptimal.png" alt="Logo" width="250" height="250">
  </a>
</p>
<h2 align="center">BeeOptimal</h2>
<h4 align="center">A Python implementation of the Artificial Bee Colony (ABC) optimization algorithm</h4>



This repository contains a Python implementation of the Artificial Bee Colony (ABC) optimization algorithm, along with several of its variants. Final project for the *Optimization for Artificial Intelligence* course, MSc's degree in Data Science and Artificial Intelligence @ University of Trieste.


# TO-DO list

- [ ] Improvements for plotting function:
  - [ ] Add option to plot only optimal bee
  - [ ] Find a way to plot optimal solution in above layer
- [ ] Try different selection strategies/different configurations and compare.
- [ ] Directed ABC
- [ ] Constraint handling
- [ ] Add arg parser for command line usage
- [ ] Add docstrings
- [ ] Try to practice with docker and create a container for the project
- [ ] Understand better PyPy packages (-> for what i understood the best practice is to have both setup.py and pyproject.toml)

Note: for installation propose 3 ways: clone and do pip install ., clone and use poetry (should work thanks to .toml file) , directly from pip (if I publish the package)

NOTE: the n_bees assert is actually dependent on the mutation strategy....e.g. if we select ABC/best/2 we need 4 donors different from the current bee,
so we need both the onlookers and the employees to be at least 5. As for now i set a unique assert for the number of bees, but it should be more flexible.
I should assert the number of bees for each configuration (4 cases)!


NOTE: try to manage padding for plots...it would be beneficial for reducing the time needed to make the plots...
# Installation

To install the package, you can use the following command:

```bash
pip install .
```

After this you can import the package in your Python code using:

```python
import beeoptimal
from beeoptimal import ArtificialBeeColony,Bee
# or
from beeoptimal.abc import ArtificialBeeColony
from beeoptimal.bee import Bee
from beeoptimal.benchmarks import *
from beeoptimal.plotting import ContourPlotBee
```