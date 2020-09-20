import numpy as np
from scipy.fftpack import fft, fftshift, ifft
# from input_data import sort_train_data


data_path = 'data'


char_matrix = np.array([
	['A', 'B', 'C', 'D', 'E', 'F'],
	['G', 'H', 'I', 'J', 'K', 'L'],
	['M', 'N', 'O', 'P', 'Q', 'R'],
	['S', 'T', 'U', 'V', 'W', 'X'],
	['Y', 'Z', '1', '2', '3', '4'],
	['5', '6', '7', '8', '9', '0'],
])

def get_p300_num(ch):
	idx = np.argwhere(char_matrix==ch)
	x, y = idx[0][0], idx[0][1]
	return x + 1, y + 7

def xy_get_char(x, y):
	try:
		return char_matrix[x-1][y-7]
	except:
		return '-'
#------------------------------------------------------------------



if __name__ == "__main__":
	# data = load_data()
	# X_train = data['X_train']
	import pdb; pdb.set_trace()
