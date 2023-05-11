#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x, y = np.random.multivariate_normal(mean, cov, 2000).T
y += 180

plt.scatter(x, y, color='magenta')  # Plot data as magenta points
plt.xlabel('Height (in)')  # Set x-axis label
plt.ylabel('Weight (lbs)')  # Set y-axis label
plt.title("Men's Height vs Weight")  # Set plot title
plt.show()  # Display plot
