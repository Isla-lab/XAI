
# Rocksample Domain with Policy Heuristics in POMCP

This repository contains code for solving the **Rocksample domain** using policy heuristics expressed in ASP formalism and implemented in **POMCP** (Partially Observable Monte Carlo Planning). The core solver is implemented in C++.

## Repository Structure
- **`pomcp/`**: Contains the C++ implementation of the POMCP solver. This folder needs to be built before running the solver.
- In **`pomcp/simulator.cpp`** you will find the core simulation logic, including the `SelectRandom` function you will have to complete in this lab lesson.


## Building the Solver
The solver needs to be built. A Makefile is included to help you. Navigate to the pomcp folder and follow the instructions in the INSTALL file.


## Lab Exercise

### Task Overview
The `simulator.cpp` file includes a partially implemented function, `SelectRandom`. Your task is to complete this function, focusing on two key steps:

1. Implement probability calculation for selecting actions according to the (already defined) logical policy.
2. Use the calculated probabilities to sample an action.
Follow the comments in the code to get some hints.

To test your solution remember to re-build the solver and then run the `scripts/run_tests.sh` script.
