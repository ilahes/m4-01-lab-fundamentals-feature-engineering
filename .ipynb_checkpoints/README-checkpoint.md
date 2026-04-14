![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Lab | Fundamentals & Feature Engineering

## Overview

Machine learning models are only as good as the data you feed them. Before you can train a model that makes useful predictions, you need to understand your dataset deeply and transform raw features into a format that algorithms can learn from effectively.

In this lab, you'll work with the classic Titanic dataset to practice the full pre-modeling workflow: exploring data, engineering new features, scaling numeric variables, and reducing dimensionality. These are the foundational skills you'll use in every ML project going forward — get comfortable with them now, and the modeling steps later will feel much smoother.

By the end, you'll have a clean, well-engineered feature set ready for classification — and a solid intuition for why these preprocessing steps matter so much.

## Learning Goals

By the end of this lab, you should be able to:

- Perform exploratory data analysis to assess data quality, distributions, and class balance.
- Engineer new features from existing columns and encode categorical variables for ML.
- Apply and compare different feature scaling techniques (StandardScaler, MinMaxScaler).
- Use PCA for dimensionality reduction and interpret explained variance.

## Setup and Context

You'll work inside a Jupyter Notebook for this lab. All analysis, code, and written interpretations should live in a single notebook so that your reasoning is visible alongside the output.

This lab directly applies the concepts from today's lesson on ML fundamentals and feature engineering. You'll use pandas for data manipulation, seaborn for loading the dataset and visualization, and scikit-learn for scaling and PCA.

## Requirements

### Fork and clone

1. Fork this repository to your own GitHub account.
2. Clone the fork to your local machine.
3. Navigate into the project directory.

### Python environment

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Getting Started

1. Create a new Jupyter Notebook called **`m4-01-fundamentals-feature-engineering.ipynb`**.
2. Start with an import cell:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA

sns.set_style("whitegrid")
```

3. Work through the tasks in order. Each task builds on the previous one.
4. Include markdown cells between code cells to explain your observations and reasoning.

## Tasks

### Task 1: Data Exploration

Load the Titanic dataset from seaborn and get familiar with its structure.

```python
titanic = sns.load_dataset("titanic")
```

1. Display the first few rows and inspect the shape of the dataset.
2. Check data types for each column. Which are numerical? Which are categorical?
3. Identify missing values — which columns have them, and what percentage of values are missing?
4. Examine the class balance of the target variable (`survived`). What is the survival rate?
5. Create at least two visualizations: one showing the distribution of a numerical feature (e.g., `age`) and one showing survival rates broken down by a categorical feature (e.g., `pclass` or `sex`).

Write a brief summary of your findings in a markdown cell.

### Task 2: Feature Engineering

Transform the raw features into a format suitable for machine learning.

1. **Handle missing values:**
   - Fill missing `age` values with the median age.
   - Fill missing `embarked` values with the mode (most frequent value).
   - Drop the `deck` column (too many missing values to be useful).

2. **Encode categorical features:**
   - Apply label encoding to `sex` (male → 1, female → 0).
   - Apply one-hot encoding to `embarked` (creating dummy columns).

3. **Create new features:**
   - `family_size` = `sibsp` + `parch` + 1 (the passenger plus their siblings/spouses and parents/children).
   - `is_alone` = 1 if `family_size` == 1, else 0.

4. Drop non-numeric and identifier columns that won't be useful for modeling (e.g., `alive`, `who`, `adult_male`, `embark_town`, `class`, `alone`).

5. Display the final DataFrame and confirm all columns are numeric.

### Task 3: Feature Scaling & Selection

Apply scaling techniques and identify the most relevant features.

1. **Compare scalers:** Apply both `StandardScaler` and `MinMaxScaler` to the numeric features. Plot the distributions of `age` and `fare` before and after each scaling method (use histograms or KDE plots). In a markdown cell, explain the difference between the two approaches.

2. **Correlation analysis:** Compute the correlation matrix for all numeric features. Plot it as a heatmap with annotations. Which features have the strongest positive and negative correlations with `survived`?

3. **Feature selection:** Based on the correlation matrix, select the top features most correlated with the target (absolute correlation ≥ 0.1). List them and briefly justify keeping or dropping borderline features.

### Task 4: Dimensionality Reduction

Apply PCA to explore the dataset in reduced dimensions.

1. Standardize all numeric features (use `StandardScaler` — PCA requires standardized data).
2. Fit PCA with all components. Plot the **explained variance ratio** for each component as a bar chart, and plot the **cumulative explained variance** as a line chart. How many components do you need to capture at least 80% of the variance?
3. Refit PCA with 2 components. Create a scatter plot of the first two principal components, colored by `survived`. Can you see any separation between survivors and non-survivors?
4. In a markdown cell, discuss: Does PCA help separate the classes? Would you use PCA as a preprocessing step for this dataset, or is the original feature space sufficient?

## Submission

### What to submit

- `m4-01-fundamentals-feature-engineering.ipynb` — your completed notebook with all code, outputs, and markdown explanations.

### Definition of done (checklist)

- [ ] Titanic dataset is loaded and explored with summary statistics and visualizations.
- [ ] Missing values are handled and categorical features are encoded.
- [ ] New features (`family_size`, `is_alone`) are created.
- [ ] Both StandardScaler and MinMaxScaler are applied and compared visually.
- [ ] Correlation matrix is plotted and top features are identified.
- [ ] PCA is applied, explained variance is plotted, and 2D projection is visualized.
- [ ] Markdown cells explain your reasoning at each step.
- [ ] The notebook runs top-to-bottom without errors (`Kernel → Restart & Run All`).

### How to submit (Git workflow)

```bash
git add .
git commit -m "lab: complete fundamentals and feature engineering"
git push origin main
```

Then open a **Pull Request** on the original repository with a brief description of your work.
