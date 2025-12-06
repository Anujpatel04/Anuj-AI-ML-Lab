# Supervised Learning Algorithms

This directory contains implementations of various supervised learning algorithms. Each algorithm is implemented as a standalone script that you can run to see how it works in practice.

## What's Here

Currently, this folder includes implementations of:

- **Linear Regression** - Basic linear regression using the California housing dataset
- **Polynomial Regression** - Polynomial regression with degree 2, also using the California housing dataset
- **Logistic Regression** - Binary classification using logistic regression (converts the housing dataset to a classification problem)

All implementations use scikit-learn and include visualization to help you understand how the algorithms perform.

## Running the Algorithms

Each script is self-contained and can be run directly:

```bash
python3 LinearRegression.py
python3 PolynomialRegression.py
python3 LogisticRegression.py
```

The scripts will:
- Load and prepare the data
- Train the model
- Evaluate performance with appropriate metrics
- Display plotted graphs showing the results

Each algorithm includes visualization plots to help you understand how the model fits the data and performs on the test set.

## What's Coming

I'll be adding more supervised learning algorithms here over time. This is a work in progress, so expect to see additional implementations as I continue building out this collection.

## Requirements

The scripts use standard machine learning libraries:
- numpy
- pandas
- matplotlib
- scikit-learn

Make sure you have these installed before running any of the scripts.

