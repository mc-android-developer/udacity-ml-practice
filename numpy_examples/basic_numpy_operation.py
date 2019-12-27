#!/usr/bin/python3

import numpy

# Two matrices are initialized by value
x = numpy.array([[1, 2], [3, 4]])
y = numpy.array([[7, 8], [9, 10]])

print("X: ")
print(x)
print()

print("Y: ")
print(y)
print()

#  matrix * const = each cell of matrix multiplied to const
print("Multiply matrix to const x*2: ")
n = 2 * x
print(n)
print()

#  matrix + const = const added to each cell of matrix
print("Multiply matrix to const x+10: ")
n = 10 + x
print(n)
print()

#  add()is used to add matrices
print("Addition of two matrices: ")
print(numpy.add(x, y))
print()

# subtract()is used to subtract matrices
print("Subtraction of two matrices : ")
print(numpy.subtract(x, y))
print()

# divide()is used to divide matrices
print("Matrix Division : ")
print(numpy.divide(x,y))
print()

print("Multiplication of two matrices: ")
print(numpy.multiply(x,y))
print()

print("The product of two matrices : ")
print(numpy.dot(x,y))
print()

print("square root is : ")
print(numpy.sqrt(x))
print()

print("The summation of elements : ")
print(numpy.sum(y))
print()

print("The column wise summation  : ")
print(numpy.sum(y, axis=0))
print()

print("The row wise summation: ")
print(numpy.sum(y, axis=1))
print()

# exp(x) = pow(math.e, x)
# for each cell of matrix set it to be math.e ^ x
print("The exp(x): ")
print(numpy.exp(x))
print()

# using "T" to transpose the matrix
print("Matrix transposition: ")
print(x.T)
print()

print("Generate range(-1, 1) matrix of floats with 0.1 step: ")
N = numpy.arange(-1, 1, 0.1)
print(N)
print()

print("Generate range(-10, 10) matrix of ints with 2 step: ")
N = numpy.arange(-10, 11, 2, numpy.int16)
print(N)
print()
