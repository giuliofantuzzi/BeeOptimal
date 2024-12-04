<div align="center">
  <a href="https://github.com/giuliofantuzzi/BeeOptimal">
    <img src="assets/LogoBeeOptimal.png" alt="Logo" width="250" height="250">
  </a>
<h2 align="center">BeeOptimal</h2>
<h4 align="center">A Python implementation of the Artificial Bee Colony (ABC) optimization algorithm</h4>
</div>

This repository contains a Python implementation of the Artificial Bee Colony (ABC) optimization algorithm, along with several of its variants. Final project for the *Optimization for Artificial Intelligence* course, MSc's degree in Data Science and Artificial Intelligence @ University of Trieste.


# TO-DO list

- [ ] Improvements for plotting function:
  - [ ] Add option to plot only optimal bee
  - [ ] Find a way to plot optimal solution in above layer
- [ ] Try different selection strategies/different configurations and compare.
- [ ] Directed ABC
- [ ] Constraint handling
- [ ] Add arg parser for command line usage



# Installation

To install the package, you can use the following command:

```bash
pip install .
```

After this you can import the package in your Python code using:

```python
import bee-optimal
from bee-optimal import ArtificialBeeColony,Bee
```