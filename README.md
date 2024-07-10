# SCOD: From Heuristics to Theory (ECCV 2024) 

This repository contains research code for the research paper: **SCOD: From Heuristics to Theory** by V.Franc, J.Paplham and D.Prusa.

The code is organized into three separate repositories, each serving a distinct task necessary to reproduce the reported results.

![](SCOD-Evaluation/docs/2d-example.png)

# Folder Structure

- OpenOOD: 
    - Our fork of the OpenOOD benchmark. It is used to download the required datasets and compute scores of different methods on the data. The benchmark is used to evaluate OOD detection, however, the exported scores can also be used to evaluate performance on the SCOD problem.

- SCOD-Likelihood-Estimation:
    - Repository used to train the estimator $\hat g(x)$ of the likelihood ratio $g(x)$.

- SCOD-Evaluation:
    - Repository used to evaluate the double-score strategies and the single-score strategies on the SCOD problem.

For details, please refer to `README.md` in each of the aforementioned folders.