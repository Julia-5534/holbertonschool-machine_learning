#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y = np.arange(0, 11) ** 3

x = np.arange(0, 11)  # Create array for x-axis

plt.plot(x, y, 'r-')  # Plot y as solid red line
plt.xlim(0, 10)  # Set the x-axis limits to 0 and 10
plt.show()  # Show the plot
