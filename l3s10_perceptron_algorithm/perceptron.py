#!/usr/bin/python3

import numpy
import time

def generate_random_input(X = 2, Y = 10, min = 0, max = 10):
	values = numpy.random.randint(min, max, (X, Y))
	print('Random int matrix of size ' + str(X) + 'x' + str(Y) + ' created:')
	print(values)


class Linear2XPerceptron:
	def __init__(self, w1, w2, b):
		self.w1 = w1
		self.w2 = w2
		self.W = numpy.array([w1, w2])
		self.b = b

	def calc(self, x1, x2):
		return self.w1 * x1 + self.w2 * x2 + self.b

	def calcv(self, X):
		return numpy.dot(self.W, X) + self.b


def main():
  print('Hello Perceptron!')

  generate_random_input()

  p = Linear2XPerceptron(1, 2, 3)

  res = p.calc(1,3)
  print(res)

  res = p.calcv(numpy.array([1, 3]))
  print(res)

  
if __name__== '__main__':
  main()
