# A comparative analysis of optimization algorithms for activity-based applications

## Introduction

This is an Constraint Programming adaptation of the econometric Activity-Based Scheduling model developed by 
[Pougala et al. (2021)](https://transp-or.epfl.ch/php/abstract.php?type=1&id=PougHillBier21).
These models build a daily schedule for an agent given a list of possible activities and their
parameters. The models are expressed as a set of constraints and an objective function to maximise.

We compare the computational performance of CP and MILP activity-based models and find that CP speeds up computation
significantly, especially as the number of activities increases. More details can be found in the project 
report:
- [Semester project report: A comparative analysis of optimization algorithms for activity-based applications](cp_abm_report.pdf)

## Code structure

We take the original MILP from the paper (located in `code/milp`) and define three new models in constraint programming:
 - `code/cp/model_basic.py`: a direct CP translation of the MILP
 - `code/cp/model_indexed.py`: an adaptation of the basic model that uses array indexing constraints instead of activity
duplication for mode and location choice
 - `code/cp/model_interval.py`: same as the previous model but adds interval variables for activity time spans

All models are defined using the [Google OR-Tools](https://developers.google.com/optimization) CP-SAT constraint solver.

Our models use the code in the following files:
- `code/cp/parameters.py`: data preparation and error function generation
- `code/cp/schedules.py`: extracting and visualizing solved schedules
- `code/cp/utils.py`: time scaling and piecewise/stepwise utilities

## How to run

### Environment and dependencies

We use the `conda` package from Anaconda to manage dependencies. Once Anaconda is installed on your local computer, we 
you can install all the dependencies using this command:

```conda env create -f environment.yml```

Additionally, to run the MILP, `IBM ILOG® CPLEX® Optimization Studio` must be installed on the computer. 
The `docplex` python bindings should be installed with the conda environment.

### Running the models

You can run the comparisons between the models using:
 
```python code/runner.py```

You can choose which set of activities use in `runner.py:main()`. New datasets can be added in the `res/` folder. To use
them, create a load function in `generation.py` and import it in `runner.py`.

You can configure which models are compared by changing the `runner.py:compare()` function.

Results and generated schedules are save to the `out/` folder, first by dataset then by model.

## About

This project is a semester research project I did at the EPFL
[Transport and Mobility laboratory](https://www.epfl.ch/labs/transp-or/), as part of my Masters in 
Computer Science.

This project was done under the supervision of [Janody Pougala](https://transp-or.epfl.ch/personnal.php?Person=POUGALA),
[Tom Häring](https://transp-or.epfl.ch/personnal.php?Person=HAERING), and 
[Prof. Michel Bierlaire](https://people.epfl.ch/michel.bierlaire). Their help and advice was invaluable during this project, I am 
grateful for their supervision.