#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

plt.hist(student_grades, bins=range(0, 101, 10), edgecolor='black')
plt.xlabel('Grades')
plt.ylabel('Number of Students')
plt.title('Project A')

# Set x and y axis ticks and limits
plt.xticks(np.arange(0, 101, 10))
plt.yticks(np.arange(0, 31, 5))
plt.xlim(0, 100)
plt.ylim(0, 30)

plt.show()
