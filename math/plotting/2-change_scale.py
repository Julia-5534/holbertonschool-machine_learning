#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 28651, 5730)
r = np.log(0.5)
t = 5730
y = np.exp((r / t) * x)

plt.plot(x, y, color='blue')  # Plot data as a red line
plt.xlabel('Time (years)')  # Set x-axis label
plt.ylabel('Fraction Remaining')  # Set y-axis label
plt.title('Exponential Decay of C-14')  # Set plot title
plt.yscale('log')  # Set y-axis scale to logarithmic
plt.xlim(0, 28650)  # Set x-axis range
plt.show()  # Display plot
