# MICRO-515 Final Project — Generalist Robot via Specialist Combination

### Group E - Paul Huot-Marchand, Edgar Colin



## Approach Overview

Our strategy is to evolve three terrain-specific **specialist robots** and combine them into a single **generalist** using multi-objective optimization, rather than training a generalist from scratch.

---

## Step 1 — Flat Specialist (560-param genome)

Starting from the HW1 ant genome (560 parameters: neural network weights + 4 body segment size parameters), we ran **CMA-ES** optimisation on the flat environment. A directional penalty was added to the fitness function to correct drift and ensure the robot moves **straight along the x-axis**.

Controller: `NeuralNetworkController(input_size=27, output_size=8, hidden_size=16)`

---

## Step 2 — Ice Specialist

Using the flat specialist genome as a warm start (pre-trained initialisation), we ran a second **CMA-ES** pass on the ice environment. Warm-starting from the flat genome allowed fast adaptation while preserving locomotion structure.

---

## Step 3 — Hill Specialist

Same warm-start strategy from the flat genome, this time optimising on the hill environment. Sharing a common base genome with the flat specialist was intentional: it maximises genome overlap across specialists, which benefits recombination in the next step.

---

## Step 4 — NSGA-II Multi-Objective Combination

With three specialists in hand, we ran **NSGA-II** with three simultaneous objectives (one per environment). The initial population was seeded using `best_genome_selection.py`, which samples diverse individuals from the top-K genomes of each specialist run to inject **structured diversity** rather than random noise.

At this stage, **4 additional body parameters** (segment sizes) were added to the genome and optimised with an independent σ₀, separate from the pre-trained controller weights.

---

## Step 5 — Selecting the Generalist

Two selection strategies were evaluated on the Pareto front:

- **Maximin**: selects the individual with the highest  score across the worst-performing environments.
- **Mean-2-worst**: selects the individual with the highest average score across the two worst-performing environments.


The **mean-to-worst** genome was retained as the final submission.
---

## Step 6 — Final CMA-ES Fine-Tuning (inconclusive)

A final CMA-ES pass was run on the composite evaluation environment, initialised from the mean-to-worst genome. No consistent improvement was observed; the genome likely reached a local optimum. The **NSGA-II mean-to-worst genome** is therefore used as the final submission.

---

## Files

| File | Description |
|---|---|
| `train_specialist.py` | CMA-ES Traning |
| `final_project_train.py` | Used for NSGA2 |
| `Robot.xml` | Final generalist body parameters|
| `x_best.npy` | Final generalist genotype (560)|
| `mlp.py` | Controller (hidden_size=16) |
| `README.md` | This file |
