import numpy


def generate_random_input(X=10, Y=2, min=0, max=10):
	res = numpy.random.uniform(min, max, (X, Y))
	print('Random int matrix of size ' + str(X) + 'x' + str(Y) + ' created:')
	print(res)
	print()
	return res


def label_input_data(data):
	s = data.shape
	if s[1] != 2:
		raise Exception('Bad shape ' + str(s) + ' Input data size must be Nx2')

	res = list()
	for i in data:
		# Labling logic can be adjusted here
		score = i[0] * 3 + i[1] * 2 + 5
		if score > 2:
			res.append(1)
		else:
			res.append(0)

	res = np.array(res)
	print('Input data labels: ')
	print(res)
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
