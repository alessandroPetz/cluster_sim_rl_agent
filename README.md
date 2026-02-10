# ⚡ Reinforcement Learning for Dynamic Power cap management

This project implements a **Reinforcement Learning (RL)** agent for intelligent **power cap management** in an HPC (High-Performance Computing) cluster simulator.  
The goal is to minimize energy consumption while maintaining system performance, by dynamically allocating power to compute nodes according to workload demand.

---

## Experiment 1
The RL agent is **trained on one workloads** and **tested on the same workload**, in order to evaluate its **capability of doing better than a HARD powercap rule**.

## Experiment 2
The RL agent is **trained on five workloads** (`1–5`) and **tested on a sixth workload (`6`)** that it has **never seen before**, in order to evaluate its **generalization capability**.

To test the agent, you need to download the HPC cluster simulator and the input files. Don’t worry, a ready-to-use Docker container can be provided to you.

for more info write at alessandro.petrella@unibo.it