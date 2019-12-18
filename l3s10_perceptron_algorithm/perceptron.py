#!/usr/bin/python3

import numpy
import time

def generate_random_input(X = 10, Y = 2, min = 0, max = 10):
	res = numpy.random.randint(min, max, (X, Y))
	print('Random int matrix of size ' + str(X) + 'x' + str(Y) + ' created:')
	print(res)
	return res

class Linear2XPerceptron:
	def __init__(self, w1 = 1, w2 = 1, b = 0):
		self.config(w1, w2, b)

	def config(self, w1, w2, b):
		self.w1 = w1
		self.w2 = w2
		self.W = numpy.array([w1, w2])
		self.b = b

	def calc2(self, x1, x2):
		return self.w1 * x1 + self.w2 * x2 + self.b

	def calc(self, X):
		return numpy.asscalar(numpy.dot(self.W, X) + self.b)

def main():
  print('Hello Perceptron!')

  x1x2 = generate_random_input()

  p = Linear2XPerceptron()
  p.config(1, 2, 3)

  for i in x1x2:
  	print('For input ' + numpy.array2string(i) + ' perceptron result is ' + str(p.calc(i)))

if __name__== '__main__':
  main()
