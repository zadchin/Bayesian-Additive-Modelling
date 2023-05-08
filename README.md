# Statistics 220 Final Project

This code repository contains code which is written for the final project for Statistics 220: Bayesian Data Analysis, a class both authors are enrolled in under the instruction of Professor Jun Liu and Teaching Fellow Buyu Lin at Harvard University.

# Table of Contents
1. [Introduction](#introduction)
2. [Models Considered](#models-considered)
3. [Dataset Considered](#dataset-considered)
4. [Evaluation Metrics](#evaluation-metrics)
5. [Code Repository Structure](#code-repository-structure)
6. [Installation Guide](#installation-guide)
7. [Papers & Slides](#link-to-paper--slides)

# Introduction

Having studied the theory of additive modeling in depth in the paper, both from a frequentist and a Bayesian perspective, we want to apply special cases of the models we have introduced in various empirical investigations. The prime goal of this data analysis, beyond understanding and experiencing how such AMs are implemented in practice, is to find out how the different variations of the same additive modeling approach perform when benchmarked against the same type of data, as well as to figure out which properties of the data yield these differences in model performance.

This code repositories contains all code used for empirical investigations.

# Models Considered

We have implemented nine models for our empirical analysis, namely:

-  Linear Model: we consider linear regression as a simple benchmark model, running the ordinary least squares method. Since all three datasets are nonlinear, we expect this model to perform poorly in comparison to the nonlinear, and whence more flexible, models in our list.

- Random Forest: we consider a random forest as a more sophisticated but less interpretable model for benchmarking purposes.

- Frequentist Additive Model: we consider a model using B-splines, which is a more general case of the cubic splines, under the penalized regression framework using second derivative smoothing as our penalty.

-  Squared Exponential Kernel Gaussian Process Prior: we consider a Gaussian process prior model introduced from Section 3 and Subsection 3.3 in our paper, using the squared exponential kernel function $k_{\text{SE}}(x,x')$. We use the eigendecomposition method detailed in Subsection 3.3 in the paper.

- Rational Quadratic Kernel Gaussian Process Prior: we consider a Gaussian process prior model introduced from Section 3 and Subsection 3.3 in our paper, using the rational quadratic kernel function $k_{\text{RQ}}(x,x')$. Similarly, we use the eigendecomposition method detailed in Subsection 3.3 in the paper.

- Ornstein-Uhlenbeck Kernel Gaussian Process Prior: we consider a Gaussian process prior model introduced from Section 3 and Subsection 3.3 in our paper, with the Ornstein-Uhlenbeck kernel function $k_{\text{OU}}(x,x')$. We use similar eigendecomposition methods as above.

- Difference Priors with Smooth Functions: we consider a model using B-splines and one-dimensional difference priors, as outlined in Section 4 and Subsection 4.1. Here, we only have the smoothness prior as our choice of difference prior.

- Difference Priors with Periodic Functions: we consider a model using B-splines and one-dimensional difference priors, as outlined in Section 4 and Subsection 4.1. Here, we will incorporate some periodicity into the difference prior, but in different ways. For the first two datasets, we consider both a smooth and a periodic component, while in the last dataset, we use only a periodic prior. 

- Difference Priors with Mixture of Functions: we consider a model using B-splines and one-dimensional difference priors, as outlined in Section 4 and Subsection 4.1. Here, we will incorporate a mixture of properties into the difference prior depending on the dataset. For the first two datasets, we consider a smooth plus periodic plus symmetric mixture, while in the last dataset, we only use a smooth and periodic mixture as symmetry is not a reasonable assumption.

# Datasets Considered

We have considered 3 datasets in our analysis, namely:

- Synthetic Functions: 50 datapoints sampled with scaled Normal noise around the true function $y=-\sin^3(x)+\cos^3(x)$, for $x\in [-\pi,\pi]$.
- mcycle Dataset: a dataset introduced by Silverman, where we examine the relationship between time and the acceleration of one's head in a simulated motorcycle accident. This is often used to test the robustness and efficacy of crash helmets.
- Beijing Air Quality Dataset: a dataset sourced from UCI Machine Learning Repository and first introduced in a 2017 paper by Zhang et al.. The dataset includes hourly air pollutants data from 12 nationally-controlled air quality sites. This data is collected by the Beijing Municipal Environmental Monitoring Center, and the meteorologial data in each air quality site is compared with the nearest weather station from the China Meteorological Administration. This data was collected from March 1, 2013 to February 28, 2017.

# Evaluation Metrics

We applied four different statistics, namely the mean squared error (MSE), the root-mean-square error (RMSE), the mean absolute error (MAE), and the coefficient of determination ($R^2$) to measure performance after model fitting.


# Code Repository Structure

**Code Files**
- paper_figures.ipynb: Python notebook detailing code for all figures generated in the paper.
- synthetic.ipynb: Python notebook detailing fitting of a synthetic function $-\sin^3(x)+\cos^3(x)$.
- synthetic.py: contains helper functions to synthetic.ipynb.
- myclce.ipynb: Python notebook detailing the fitting of the mycycle dataset.
- mycycle.py: contains helper functions for mycycle.ipynb.
- beijing_EDA.ipynb: Python notebook detailing an exploratory data analysis process on the Beijing Multi-Site Air-Quality dataset.
- beijing.ipynb: Python notebook detailing the fitting of the Beijing Multi-Site Air-Quality dataset.
- utils.py: contains functions for the interpolation of 1d and 2d matrices, as well as code for smooth functions for the fitting of difference prior AM models.

**Folders**
- data: contains the original multivariate data from the Beijing Multi-Site Air-Quality Dataset, specifically the one collected at Aoti Zhongxin, from 2013-2017. 
- paper_figures: contains figures from paper_figures.ipynb.
- EDA_figures: contains figures from beijing_EDA.ipynb.
- synthetic_figures: contains figures from synthetic.ipynb.
- mycycle_figures: contains figures from mycycle.ipynb.
- model_figures: contains figures from beijing.ipynb.

# Installation Guide

- Clone the github file to your local desktop; In your terminal, run `git clone https://github.com/zadchin/Bayesian-Additive-Modelling.git`
- Check and install requirements.txt: `pip install -r requirements.txt`
- Assuming you have Python and Conda configured as well as a text editor, run code files.

# Link to Paper & Slides

The paper can be accessed at under docs folder at [here](docs/Stat220FinalProject.pdf)

The slides can be accessed at under docs folder at [here](docs/Stat220FinalPresentation.pdf)