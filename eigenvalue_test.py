import time
import numpy as np
import h5py
import fileinput
import sys
import glob
import scipy.linalg

eps = 1e-16
a = 1
b = .5 + eps

A = np.zeros((2,2))
A[0,0] = a
A[1,1] = 0
A[0,1] = b
A[1,0] = -b
print("matrix = ")
print(A)

eigvals = scipy.linalg.eigvals(A)
print("eigvals = ",eigvals)
print("imaginary part should be ",np.sqrt(eps))


print()
print("==========")
print()
a=1e-15
b=1e-18
A[0,0] = a
A[1,1] = a
A[0,1] = b
A[1,0] = -b
print("matrix = ")
print(A)

eigvals = scipy.linalg.eigvals(A)
print("eigvals = ",eigvals)
print("imaginary part should be ",b)


