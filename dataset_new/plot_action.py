import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Read the file and extract data
file_path = '/home/olagh/RLBench/dataset_new/actions.txt'

# Assuming each line contains x, y, z coordinates separated by space or comma
data = np.loadtxt(file_path, delimiter=',')  # Adjust delimiter if needed

# Step 2: Extract x, y, z coordinates
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]

# Step 3: Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
ax.scatter(x, y, z, c='r', marker='o')

# Labels
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# Show plot
plt.show()
