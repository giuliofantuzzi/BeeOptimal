<div align="center">
  <a href="https://github.com/giuliofantuzzi/BeeOptimal">
    <img src="assets/LogoBeeOptimal.png" alt="Logo" width="250" height="250">
  </a>
<h2 align="center">BeeOptimal</h2>
<h4 align="center">A Python implementation of the Artificial Bee Colony (ABC) optimization algorithm</h4>
</div>

This repository contains a Python implementation of the Artificial Bee Colony (ABC) optimization algorithm, along with several of its variants. Final project for the *Optimization for Artificial Intelligence* course, MSc's degree in Data Science and Artificial Intelligence @ University of Trieste.


# TO-DO list
- [x] Implement original version of ABC algorithm
- [x] Tests with some benchmark functions
- [x] Plots with colony convergence
- [ ] Improvements for plotting function:
  - [ ] Add option to plot only optimal bee
  - [ ] Find a way to plot optimal solution in above layer
- [ ] Refactor ABC code (mainly redundant attributes)
- [ ] Think about additional termination criteria
- [ ] Try different selection strategies/different configurations and compare

NOTE: instead of comparing ABC with other algorithms (GA,DE,PSO,etc) I think it is more interesting to
compare different versions of ABC. Why such decision? Implementing other algorithms would be a lot of work, and each technique has its own peculiarities and its own parameters to tune. ABC and its variants
instead have some common parameters, so we can perform a more fair and consistent comparison
