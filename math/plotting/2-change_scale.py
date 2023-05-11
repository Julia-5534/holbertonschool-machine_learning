#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 28651, 5730)
r = np.log(0.5)
t = 5730
y = np.exp((r / t) * x)

plt.plot(x, y, color='blue')  # Plot the data as a red line
plt.xlabel('Time (years)')  # Set the x-axis label
plt.ylabel('Fraction Remaining')  # Set the y-axis label
plt.title('Exponential Decay of C-14')  # Set the plot title
plt.yscale('log')  # Set the y-axis scale to logarithmic
plt.xlim(0, 28650)  # Set the x-axis range
plt.show()  # Display the plot
