# ⚡ Reinforcement Learning for Dynamic Power cap management

This project implements a **Reinforcement Learning (RL)** agent for intelligent **power cap management** in an HPC (High-Performance Computing) cluster simulator.  
The goal is to minimize energy consumption while maintaining system performance, by dynamically allocating power to compute nodes according to workload demand.

---

## Experiment 1
The RL agent is **trained on one workloads** and **tested on the same workload**, in order to evaluate its **capability of doing better than a HARD powercap rule**.


The RL agent is **trained on four workloads** (`1–4`) and **tested on a fifth workload (`5`)** that it has **never seen before**, in order to evaluate its **generalization capability**.


