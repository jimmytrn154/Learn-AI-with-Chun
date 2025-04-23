import numpy as np
import math


# def dense(a_in, W, b):
#     units = W.shape[1]
#     a_out = np.zeros(units)
#     for j in range(units):
#         w = W[:, j]
#         z = np.dot(w, a_in) + b[j]
#         a_out[j] = g(z)
#     return a_out

# W = np.array([
#     [1, -3, 5],
#     [2, 4, -6]
# ])

# b = np.array([-1, 1, 2])
# print(b[0])
# a_in = np.array([-2, 4])

# print(dense(a_in, W, b))

# def dense2(A_in, W, B):
#     Z = np.matmul(A_in, W) + B
#     A_out = g(Z)
#     return A_out

def g(z):
    return 1 / (1 + np.exp(-z))  # More stable with NumPy
AT = np.array([[200, 17]])
W = np.array([[1,-3,5], 
            [-2,4,-6]])
B = np.array([[-1,1,2]])
Z = np.matmul(AT, W) + B
print(Z)
print(g(Z))