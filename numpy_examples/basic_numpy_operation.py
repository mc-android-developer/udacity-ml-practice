#!/usr/bin/python3

import numpy

# Two matrices are initialized by value
x = numpy.array([[1, 2], [4, 5]])
y = numpy.array([[7, 8], [9, 10]])

print("X: ")
print(x)
print()

print("Y: ")
print(y)
print()

#  add()is used to add matrices
print("Addition of two matrices: ")
print(numpy.add(x,y))
print()

# subtract()is used to subtract matrices
print("Subtraction of two matrices : ")
print(numpy.subtract(x,y))
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
print(numpy.sum(y,axis=0))
print()

print("The row wise summation: ")
print(numpy.sum(y,axis=1))
print()

# using "T" to transpose the matrix
print("Matrix transposition : ")
print(x.T)
print()
