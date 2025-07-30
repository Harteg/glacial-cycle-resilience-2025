# Glacial Cycle Resilience

This repository contains the code for the paper:

**Quantifying Resilience in Non-Autonomous and Stochastic Earth System Dynamics with Application to Glacial-Interglacial Cycles**  
Jakob Harteg et al., *in review at Earth System Dynamics* (2025)

## Overview

This code implements a conceptual model of Earth system dynamics to investigate how glacial–interglacial climate trajectories respond to perturbations. Two complementary resilience metrics are introduced:
- **Reference Adherence Ratio (RAR)**: fraction of ensemble members that remain near the reference trajectory
- **Return time**: time for a single perturbed trajectory to return to the reference path

Simulations reveal strong temporal variations in system resilience across the glacial cycle.

## Repository contents

- `model.py` – main implementation of the stochastic climate model  
- `plotting.py` – helper functions to reproduce the paper’s figures  
- `plots.ipynb` – Jupyter notebook that generates all main figures from the paper

## Installation

Clone the repository and install the required Python packages:

```bash
git clone https://github.com/harteg/glacial-cycle-resilience-2025.git
cd glacial-cycle-resilience-2025
pip install -r requirements.txt