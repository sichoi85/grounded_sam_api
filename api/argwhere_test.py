import numpy as np

# Example 2D boolean array
bool_array = np.array([[[False, True, False],
                       [True, False, True],
                       [False, False, True]]])

# Convert True values to indices using np.argwhere
bool_array = bool_array[0,:,:]
index_array = np.argwhere(bool_array & ~np.roll(bool_array, 1, axis=(0, 1)))

print("Boolean Array:")
print(bool_array)

print("\nIndex Array:")
print(index_array)