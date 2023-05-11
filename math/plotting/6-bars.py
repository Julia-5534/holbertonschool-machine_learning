#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

# Define colors for each fruit
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

# Create stacked bar chart
plt.bar(np.arange(3), fruit[0], width=0.5,
        color=colors[0], label='apples')
plt.bar(np.arange(3), fruit[1], width=0.5,
        color=colors[1], bottom=fruit[0], label='bananas')
plt.bar(np.arange(3), fruit[2], width=0.5,
        color=colors[2], bottom=fruit[0]+fruit[1], label='oranges')
plt.bar(np.arange(3), fruit[3], width=0.5,
        color=colors[3], bottom=fruit[0]+fruit[1]+fruit[2], label='peaches')

# Add legend, labels, and title
plt.legend()
plt.xticks(np.arange(3), ['Farrah', 'Fred', 'Felicia'], fontsize='x-small')
plt.ylabel('Quantity of Fruit')
plt.yticks(np.arange(0, 81, 10), fontsize='x-small')
plt.title('Number of Fruit per Person', fontsize='x-small')

# Show plot
plt.show()
