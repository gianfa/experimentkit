""" Reporting example

Linear Regression

"""
# %% Imports
from pathlib import Path
import random

import numpy as np
import matplotlib.pyplot as plt

from experimentkit.reporting import ReportMD

# %% Experiment Parameters

EXP_NAME = "exp-1"
REPORT_PATH = Path(__file__).parent  # the folder of this script

# initialize the report
report = ReportMD(
    md_path=REPORT_PATH/f"{EXP_NAME}-trial_1.md",
    title=f"Report: {EXP_NAME}")

# %% 

# Generate data
x = [1, 2, 3, 4, 5]
y = [2, 3, 4, 5, 6]
noise_factor = 0.5
y = [yi + random.uniform(-noise_factor, noise_factor) for yi in y]
report.add_txt("1. Generate some data")  # it prints a new line by default

# Regression
n = len(x)
sum_x = sum(x)
sum_y = sum(y)
sum_xy = sum(x[i] * y[i] for i in range(n))
sum_x_squared = sum(x[i] ** 2 for i in range(n))

m = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)
b = (sum_y - m * sum_x) / n
predicted_y = [m * xi + b for xi in x]
report.add_txt("2. Perform the linear regression") 

report.add_txt("## Methods\n")
report.add_txt("Line equation:\n$Y_{pred} = mX + b$")

# Plot
fig, ax = plt.subplots()
ax.scatter(x, y, label='data')
ax.plot(x, predicted_y, color='red', label='LinReg')
ax.set(
    title='Linear Regression',
    xlabel='X',
    ylabel='Y',
)
ax.legend()
ax.grid(True)
fig.show()
fig.savefig(REPORT_PATH/"linreg.png")
report.add_txt("## Results\n")
report.add_img(REPORT_PATH/"linreg.png")

# Now check the created .md file and see the result!

# %%
