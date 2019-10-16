'''Plot data generated on 4x4, 5x5, 6x6, or 7x7 grids.
Save 200 DPI plot in file plotname
'''

# Author: Avisek Gupta


import numpy as np
import matplotlib.pyplot as plt

plotname = 'res_plt1.png'

# This is used to get a random colour
list_color = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
    'E', 'F'
]

# User Input : Folder number
dataset_folder_number = -1
while dataset_folder_number not in [4, 5, 6, 7]:
    print('Enter dataset folder number (4/5/6/7) :')
    dataset_folder_number = int(input())

# User Input : Dataset number
dataset_number = -1
while dataset_number not in range(1, 1000+1):
    print('Enter dataset number (1-1000) :')
    dataset_number = int(input())

# Load the data
data = np.loadtxt(
    'clusters'+str(dataset_folder_number)+'/dataset'+str(dataset_number)
)

for iter1 in range(dataset_folder_number*dataset_folder_number):
    # Get a random colour
    color1 = '#'
    for i in range(6):
        color1 = color1 + list_color[np.random.randint(16)]
    # Plot the data
    indices = (data[:, 2] == iter1)
    plt.scatter(data[indices, 0], data[indices, 1], c=color1, marker='x')

# Plot and display or save
plt.axis(
    [np.amin(data[:, 0])-1, np.amax(data[:, 0])+1,
        np.amin(data[:, 1])-1, np.amax(data[:, 1])+1]
)
plt.savefig(plotname, dpi=200)
plt.show()
