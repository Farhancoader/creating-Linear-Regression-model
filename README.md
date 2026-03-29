# creating-Linear-Regression-model
# Linear Regression from Scratch

## Overview
This project implements Linear Regression from scratch using Gradient Descent, without relying on built-in machine learning models. The goal was to understand the underlying mathematics and optimization process behind linear models.

---

## Mathematical Formulation

The model assumes a linear relationship between input features and the target:

y = Xw + b

where:
- X ∈ ℝⁿˣᵈ is the feature matrix  
- w ∈ ℝᵈ are the weights  
- b ∈ ℝ is the bias  

The objective is to minimize the Mean Squared Error (MSE):

J(w, b) = (1/n) Σ (yᵢ - ŷᵢ)²

To minimize this loss, Gradient Descent is used with the following updates:

w = w - α ∂J/∂w  
b = b - α ∂J/∂b  

where α is the learning rate.

---

## Features
- Implementation of Linear Regression using Gradient Descent
- Support for multiple input features
- Automatic handling of NumPy arrays and Pandas DataFrames
- Train-test split for evaluation
- Feature scaling for improved convergence
- Evaluation using R² score and Mean Squared Error
- Comparison with Scikit-learn implementation

---

## Results

| Model              | R² Score |
|-------------------|---------|
| Custom Model      | ~0.33   |
| Scikit-learn Model| ~0.45   |

The difference is primarily due to optimization: this implementation uses iterative gradient descent, whereas Scikit-learn uses optimized solvers or closed-form solutions.

---

## Visualizations
- Loss vs Epochs (to monitor convergence)
- Actual vs Predicted values (to evaluate performance)

---

## Technologies Used
- NumPy
- Matplotlib
- Scikit-learn (for comparison)

---

## How to Run

```bash
pip install -r requirements.txt
python train.py
