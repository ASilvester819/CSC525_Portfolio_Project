# CSC525\_Portfolio\_Project



\# Portfolio Allocation Agent 



This project trains a TensorFlow forecaster on daily industry portfolio returns and selects a long-only portfolio allocation based on a user-selected risk category.



\# What it does

\- Downloads the latest Ken French 49-Industry \*\*Daily\*\* dataset at runtime

\- Trains (or loads) a TensorFlow next-day multi-output model

\- Defines risk as: \*\*P(rolling 252-trading-day compounded return <= 0)\*\*

\- Prompts the user for a risk category (Conservative / Risk Averse / Moderate / Aggressive)

\- Outputs an optimal long-only portfolio allocation



\# How to run (Windows / Mac / Linux)



\# 1) Clone

```bash

git clone <REPO\_URL>

cd <REPO\_FOLDER>



