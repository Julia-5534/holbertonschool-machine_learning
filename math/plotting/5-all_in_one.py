#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# First plot
y0 = np.arange(0, 11) ** 3
plt.subplot(3, 2, 1)
plt.plot(y0, color='red')
plt.xticks(np.arange(0, 11, 2), fontsize='x-small')
plt.yticks(fontsize='x-small')
plt.xlim(0, 10)
plt.ylim(0, 1000)

# Second plot
mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180
plt.subplot(3, 2, 2)
plt.scatter(x1, y1, color='magenta', s=5)
plt.xlabel('Height (in)', fontsize='x-small')
plt.ylabel('Weight (lbs)', fontsize='x-small')
plt.title("Men's Height vs Weight", fontsize='x-small')
plt.xticks(fontsize='x-small')
plt.yticks(fontsize='x-small')

# Third plot
x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)
plt.subplot(3, 2, 3)
plt.semilogy(x2, y2, 'b-')
plt.xlabel('Time (years)', fontsize='x-small')
plt.ylabel('Fraction Remaining', fontsize='x-small')
plt.title('Exponential Decay of C-14', fontsize='x-small')
plt.xticks([0, 10000, 20000], fontsize='x-small')
plt.yticks(fontsize='x-small')
plt.xlim(0, 28650)  # set x-axis limits

# Fourth plot
x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)
plt.subplot(3, 2, 4)
plt.plot(x3, y31, 'r--', label='C-14')
plt.plot(x3, y32, 'g-', label='Ra-226')
plt.xlabel('Time (years)', fontsize='x-small')
plt.ylabel('Fraction Remaining', fontsize='x-small')
plt.title('Exponential Decay of Radioactive Elements', fontsize='x-small')
plt.legend(loc='upper right', fontsize='x-small')
plt.xticks(np.arange(0, 21000, 5000), fontsize='x-small')
plt.yticks(fontsize='x-small')
plt.xlim(0, 20000)
plt.ylim([0,1])

# Fifth plot
np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)
plt.subplot(3, 2, (5, 6))
plt.hist(student_grades, bins=range(0, 101, 10), edgecolor='black')
plt.xlabel('Grades', fontsize='x-small')
plt.ylabel('Number of Students', fontsize='x-small')
plt.title('Project A', fontsize='x-small')
plt.xticks(range(0, 101, 10), fontsize='x-small')
plt.yticks(fontsize='x-small')

# Overall figure
plt.subplots_adjust(wspace=0.4, hspace=0.8)
plt.suptitle('All in One', fontsize='large')
plt.show()
