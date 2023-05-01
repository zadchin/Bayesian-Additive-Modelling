# STAT 220 Final Project

This code repository contains code that are written for the final project for Statistics 220: Bayesian Data Analysis, a class both authors are enrolled in under the instruction of Professor Jun Liu and Teaching Fellow Buyu Lin at Harvard University.

## Code Repository Structure

**Code Files**
- toy_sample.ipynb: Python notebook detailing fitting of a smooth function $-sin^3(x)+cos^3(x)$ with 5 different algrithms, namely linear regression, random forest, frequentist penalized AM, Gaussian Process Prior (with 3 different kernels) and Difference Priors (with three priors: smooth prior, smooth + periodic prior, smooth + periodic + symmetric prior)
- univariate.ipynb: Python notebook detailing the fitting univariate data from a Simulated
Motorcycle Accident with the 5 different algorithms similar to above.
- EDA.ipynb: Python notebook detailing exploratory data analysis process on the Multivariate data, sourced from Beijing Multi-Site Air-Quality Data Set. Code includes data cleaning, exploring different covariates and its distributions, data standardization and transformation, as well as data splitting for cross validation. Figure results are exported under EDA_figures and data result is exported under data folder.
- helpers.py: Contains helper functions to toy_example and univariate.ipynb to simplify code structure.



**Folders**
- data: Contains the original Mltivariate Data from the Beijing Multi-Site Air-Quality Data Set, specifically the one collected at Aoti Zhong Xin, from 2013-2017. It also contains X_train, X_test, y_train, y_test csv arise from train-test split from the dataset.
- EDA_figures: Contains figures from EDA.ipynb, which is a Python notebook detailing exploratory data analysis on the Aoti dataset.
- toy_sample_figures: Contain figures from toy_example.ipynb
- univariate_figures: Contain figures from univariate.ipynb

