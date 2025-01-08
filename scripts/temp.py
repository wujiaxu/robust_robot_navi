import numpy as np

def downsample_1d(array, target_size):
    # Generate equally spaced indices
    indices = np.linspace(0, len(array) - 1, target_size, dtype=int)
    print(indices)
    return array[indices]

def count_consecutive_zeros(binary_array):
    binary_array = np.array(binary_array)
    zeros = binary_array == 0  # Create a boolean mask for zeros
    # Use cumulative sums to count zeros, but reset the count when encountering a 1
    cumsum = np.cumsum(zeros)
    reset = np.maximum.accumulate((binary_array == 1) * cumsum)
    result = cumsum - reset
    return result

# Example usage
binary_array = [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]
result = count_consecutive_zeros(binary_array)
print(result)

print(np.pi*10/180*4)
# # Example
# array = np.random.random(720)#np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# target_size = 11

# downsampled_array = downsample_1d(array, target_size)

# print("Original array:", array)
# print("Downsampled array:", downsampled_array)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch,Circle

legend_elements = [Line2D([0], [0], color='b', lw=4, label='Line'),
                   Line2D([1], [1], marker='o', color='w', label='Scatter',
                          markerfacecolor='g', markersize=15,alpha=0.5),
                    Line2D([1], [1], marker='o', label='agent',
                          color='black', fillstyle="none",markersize=15,linestyle='None'),
                   Patch(facecolor='orange', edgecolor='r',
                         label='Color Patch'),
                         Circle((0,0), 0.3, fill=True, color="black",alpha=1,visible=True)]

# Create the figure
fig, ax = plt.subplots()
ax.legend(handles=legend_elements, loc='center')
plt.savefig("debug.png")