import numpy as np
a = np.array([[5,6,7,8], [1,2,3,4]])
# print(a.shape[0])

sample = np.arange(6).reshape(-1,2) #tells numpy to calculate ideal columns for the specified colummns
print(sample)

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
m=X_train.shape[1] # shape[0] means return the amount of rows and shape[1] means return the amount of columns for 2d array and shape[0] means return amount of columns of 1D
print(m)