#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x, y = np.random.multivariate_normal(mean, cov, 2000).T
y += 180

plt.scatter(x, y, color='magenta')  # Plot the data as magenta points
plt.xlabel('Height (in)')  # Set the x-axis label
plt.ylabel('Weight (lbs)')  # Set the y-axis label
plt.title("Men's Height vs Weight")  # Set the plot title
plt.show()  # Display the plot
