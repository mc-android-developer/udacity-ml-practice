#!/usr/bin/python3

import numpy
import time

def generate_random_input(X = 10, Y = 2, min = 0, max = 10):
	res = numpy.random.randint(min, max, (X, Y))
	print('Random int matrix of size ' + str(X) + 'x' + str(Y) + ' created:')
	print(res)
	print()
	return res

def label_input_data(data):
	s = data.shape
	if s[1] != 3:
		raise Exception('Bad shape ' + str(s) + ' Input data size must be Nx3')
	
	for i in data:
		# Labling logic can be adjusted here
		score = i[0] * 3 + i[1] * 2 + 5  
		if score > 2:
			i[2] = 1
		else:
			i[2] = 0

	print('Input data labled: ')
	print(data)
	print()
	return data


def print_visualization_data(data):
	s = data.shape
	if s[1] != 3:
		raise Exception('Bad shape ' + str(s) + ' Input data size must be Nx3')

	str_data_g0 = ''
	str_data_g1 = ''
	for i in data:
		if i[2] == 0:
			str_data_g0 += '(' + str(i[0]) + ',' + str(i[1]) + '), '
		elif i[2] == 1:
			str_data_g1 += '(' + str(i[0]) + ',' + str(i[1]) + '), '
		else:
			raise Exception('Unknown label: ' + str(i[2]))
	
	print('Visualization data: ')
	print('(use https://www.desmos.com/calculator to visualize)')
	print(str_data_g0[:-2])
	print(str_data_g1[:-2])
	print()
	return data

class Linear2XPerceptron:
	def __init__(self, w1 = 1, w2 = 1, b = 0):
		self.config(w1, w2, b)

	def config(self, w1, w2, b):
		self.w1 = w1
		self.w2 = w2
		self.W = numpy.array([w1, w2])
		self.b = b

	def calc_once(self, x1, x2):
		return self.w1 * x1 + self.w2 * x2 + self.b

	def calc(self, X):
		s = X.shape
		if s[0] != 2:
			raise Exception('Bad shape ' + str(s) + ' Input data size must be Nx2')
		
		return numpy.asscalar(numpy.dot(self.W, X) + self.b)
		

def main():
  print('Hello Perceptron!')

  x1x2 = generate_random_input(X=20, Y=3, min=-5, max=5)
  label_input_data(x1x2)
  print_visualization_data(x1x2)

  p = Linear2XPerceptron()
  p.config(1, 2, 3)

  for i in x1x2:
  	print('For input ' + numpy.array2string(i[:-1]) + ' perceptron result is ' + str(p.calc(i[:-1])))

  



if __name__== '__main__':
  main()
