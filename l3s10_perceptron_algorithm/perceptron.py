#!/usr/bin/python3

import numpy as np
import data_helper as dh

class Linear2XPerceptron:
  def __init__(self, w1 = 1, w2 = 1, b = 0):
    self.config(w1, w2, b)

  def config(self, w1, w2, b):
    self.W = np.array([w1, w2])
    self.C = np.array([w1, w2, b])
    self.b = b
    return self

  def config_vec(self, C):
    s = C.shape
    if s[0] != 3:
      raise Exception('Bad shape ' + str(s) + ' Input data size must be 1x3')
    self.config(C[0], C[1],C[2])
    return self

  def calc_once(self, x1, x2):
    return self.W[0] * x1 + self.W[2] * x2 + self.b

  def calc(self, X):
    s = X.shape
    if s[0] != 2:
      raise Exception('Bad shape ' + str(s) + ' Input data size must be Nx2')
    
    return np.asscalar(np.dot(self.W, X) + self.b)

def adjust_perceptron(prc, input, lrate = 0.1):
  s = input.shape
  if s[0] != 3 and s[1] != None:
    raise Exception('Bad shape ' + str(s) + ' Input data size must be 1x3')

  print('Adjusting perceptron:')

  print('Perceptron config ' + np.array2string(prc.C))  
  print('Learning rate ' + str(lrate))
  
  n = input
  n[2] = 1 # bias is const
  print('Input ' + np.array2string(n))
  
  m = n * lrate
  print('Correction vector ' + np.array2string(m))

  t = prc.C - m
  print('Adjusted perceptron config ' + np.array2string(t))
  prc.config_vec(t)
  
  print()
  return prc


def main():
  print('Hello Perceptron!')

  size = 20
  input_data = dh.generate_random_input(X=size, Y=3, min=-5, max=5)
  dh.label_input_data(input_data)
  dh.print_visualization_data(input_data)

  p = Linear2XPerceptron()
  p.config(1, 2, 3) # start with any random config

  convergence_cnt = 0
  while convergence_cnt < 20:
    for i in input_data:
      print('-----------------------------------------------------------------------')
      print('convergence_cnt: ' + str(convergence_cnt))

      adj_cnt = 0
      res = p.calc(i[:-1])
      score = 0 if res > 0 else 1

      if score == i[2]:
        convergence_cnt += 1
        continue

      while score != i[2]:
        convergence_cnt = 0
        print('For input ' + np.array2string(i) + ' perceptron result is ' + str(p.calc(i[:-1])) + ' and score is ' + str(score))

        adjust_perceptron(p, i, 0.1)
        res = p.calc(i[:-1])
        score = 0 if res > 0 else 1

        adj_cnt += 1
        if adj_cnt > 100:
          raise Exception('Allowed adjustments number treshold exceeded')
       
      if convergence_cnt == 0:
       break   

  print()
  print()
  print('Perceptron automatic adjustment completed')
  print('Result config ' + np.array2string(p.C))  
  


      

if __name__== '__main__':
  main()