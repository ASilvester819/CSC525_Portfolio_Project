# Machine Learning Portfolio Allocation Agent  
**TensorFlow • Forward Simulation • Downside-Risk Optimization**

---

# Overview

This project implements a **machine-learning–driven portfolio construction system** that allocates capital across U.S. equity industry sectors while explicitly controlling downside risk.

Unlike traditional mean–variance optimization, this system:

- Uses TensorFlow to forecast conditional expected returns
- Models uncertainty via bootstrapped residual simulations
- Allows users to define risk tolerance directly as a maximum acceptable one-month loss
- Constructs a long-only, diversified portfolio that satisfies a Value-at-Risk (VaR) constraint

The result is a forward-looking, risk-aware portfolio selection framework suitable for academic evaluation and reproducible execution.

---

# Key Features

- Daily data (1926–present) from the Ken French Data Library
- 49 Industry Portfolios + Risk-Free Rate
- Deep neural network (TensorFlow) for return forecasting
- Monte Carlo simulation of next-month outcomes
- VaR-style downside risk constraint
- Robust reward (utility) function
- User-driven risk input
- Single executable entry point (`main.py`)

---

# Data Sources

Data is downloaded automatically at runtime from:

- 49 Industry Portfolios (Daily) – Ken French Data Library  
- Fama–French Risk-Free Rate (Daily)  

No manual data downloads are required.

---

# Methodology

# 1. Return Forecasting (Supervised Learning)

- Inputs: rolling window of past daily returns across all assets
- Model: multi-layer feed-forward neural network
- Target: next-day return vector for all assets
- Output: conditional mean forecast of daily returns

---

# 2. Uncertainty Modeling (Residual Bootstrap)

- Model residuals are computed on the training set
- Residual vectors are sampled **row-wise** to preserve cross-asset dependence
- Simulated daily returns are constructed as:



Running the Model Locally (Step-by-Step)

These instructions explain how to run the portfolio allocation model on your own machine.

Prerequisites

You will need:
Python 3.9 or newer
Git installed
An internet connection (the model downloads data automatically)

You do not need to download any datasets manually.

Step 1: Clone the Repository

Open a terminal (Command Prompt, PowerShell, or Terminal on macOS/Linux) and run:

git clone <https://github.com/ASilvester819/CSC525_Portfolio_Project>
cd CSC525_Portfolio_Project

Step 2: Create a Virtual Environment (Recommended)

This keeps dependencies isolated.

Windows

python -m venv .venv
.venv\Scripts\activate


macOS / Linux

python3 -m venv .venv
source .venv/bin/activate


After activation, your terminal prompt should show (.venv).

Step 3: Install Dependencies

Install all required Python packages:

pip install -r requirements.txt


This may take a few minutes, especially for TensorFlow.

Step 4: Run the Model

Run the program:

python main.py


The program will:

Download the required financial datasets automatically

Load or train the machine-learning model

Prompt you for a risk tolerance input

Step 5: Enter Risk Tolerance

When prompted, enter the maximum one-month loss you are willing to tolerate, as a percentage.

Example:

Enter max 1-month loss (e.g., -10): -10


Interpretation:

“With 95% confidence, I do not want to lose more than 10% next month.”

Step 6: Review Results

The program will print:

The top feasible portfolios that satisfy the risk constraint

Detailed risk statistics for each

The recommended portfolio allocation, expressed as percentages by sector

No additional interaction is required.

Optional: Force Retraining

If you want to retrain the neural network from scratch (recommended if you modify code or settings), run:

python main.py --retrain

Optional: Running in Jupyter Notebook

If you prefer Jupyter:

Launch Jupyter:

jupyter notebook


Open a notebook in the project directory

Run:

%run main.py --retrain


The model will behave the same way as in the command line.

Troubleshooting

First run may be slow (model training + simulations)

If you encounter issues:

Ensure Python version ≥ 3.9

Ensure the virtual environment is activated

Try running with --retrain

Summary (Quick Start)

For most users:

Clone the repo

Install dependencies

Run:

python main.py


Enter a risk limit when prompted