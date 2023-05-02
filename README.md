# STAT 220 Final Project

This code repository contains code that are written for the final project for Statistics 220: Bayesian Data Analysis, a class both authors are enrolled in under the instruction of Professor Jun Liu and Teaching Fellow Buyu Lin at Harvard University.

## Code Repository Structure

**Code Files**
- toy_sample.ipynb: Python notebook detailing fitting of a smooth function $-sin^3(x)+cos^3(x)$ with 5 different algrithms, namely linear regression, random forest, frequentist penalized AM, Gaussian Process Prior (with 3 different kernels) and Difference Priors (with three priors: smooth prior, smooth + periodic prior, smooth + periodic + symmetric prior)
- univariate.ipynb: Python notebook detailing the fitting univariate data from a Simulated
Motorcycle Accident with the 5 different algorithms similar to above.
- EDA.ipynb: Python notebook detailing exploratory data analysis process on the Multivariate data, sourced from Beijing Multi-Site Air-Quality Data Set. Code includes data cleaning, exploring different covariates and its distributions, data standardization and transformation, as well as data splitting for cross validation. Figure results are exported under EDA_figures and data result is exported under data folder.
- multivariate.ipynb: Python notebook detailing fitting of a train-test splitted Aoti station air pollution data, sourced from Beijing Multi-Site Air Quality Data set. There are 11 algorithms tested, namely linear regression, random forest optimized with Bayesian Optimization, Frequentist Penalized AM, Gaussian Process Priors with 3 differnt Kernels, Difference Priors with $f(x_1,x_2)$ and $f(x_1)+f(x_2)$, as well as lastly a Bayeisan Optimized Difference Priors with $f(x_1)+f(x_2)$. MSE and MAE is calculated and saved in a dataframe for every algorithm. We plot a histogram of y and the predicted y to check the resulted distribution from prediction as well.
- helpers.py: Contains helper functions to toy_example and univariate.ipynb to simplify code structure.
- utils.py: Contains functions to interpolate for 1d and 2d matrix as well as coding smooth functions for difference priors fitting.

**Folders**
- data: Contains the original Mltivariate Data from the Beijing Multi-Site Air-Quality Data Set, specifically the one collected at Aoti Zhong Xin, from 2013-2017. It also contains X_train, X_test, y_train, y_test csv arise from train-test split from the dataset.
- EDA_figures: Contains figures from EDA.ipynb, which is a Python notebook detailing exploratory data analysis on the Aoti dataset.
- toy_sample_figures: Contain figures from toy_example.ipynb
- univariate_figures: Contain figures from univariate.ipynb
- model_figures: Contain figures from multivariate.ipynb

## Installation Guide

- Clone the github file to your local desktop; In your terminal, run `git clone https://github.com/zadchin/Bayesian-Additive-Modelling.git`
- Check and intall requirement.txt: `pip install -r requirements.txt`
- Assuming you have Python and Conda configured as well as a text editor, run code files.



