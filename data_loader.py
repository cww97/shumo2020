import numpy as np
from scipy.fftpack import fft, fftshift, ifft
from utils import data_path
from input_data import sort_train_data
from input_data import persons


def add_fre(x, num_fft=46):
	Y = fft(x, num_fft, axis=2)
	Y = np.abs(Y)
	ps = Y**2 / num_fft
	# import pdb; pdb.set_trace()
	ret = np.concatenate((x, ps), axis=2)
	return ret

def divide_data(X, y, num_val=100):
	''' total: 3600/720
	'''
	X_train = X[: -num_val]
	y_train = y[: -num_val]
	X_val = X[-num_val: ]
	y_val = y[-num_val: ]
	return X_train, y_train, X_val, y_val

def normalization(X):
	mu = X.mean((0, 1, 2))
	s2 = X.std((0, 1, 2))
	X = (X - mu) / s2
	return X, mu, s2

def load_data(path=data_path, persons=persons):
	'''
	Data format:
	Datatype - float32 (both X and Y)
	X.shape - (#samples, 1, #timepoints,  #channels)
	Y.shape - (#samples)
	'''
	X, y = sort_train_data(persons=persons)
	X = X.astype('float32')

	# X = add_fre(X)
	X, mu, s2 = normalization(X)
	X = X.transpose(0, 1, 3, 2)
	X_train, y_train, X_val, y_val= divide_data(X, y)

	return {
		'X_train': X_train,
		'y_train': y_train,
		'X_val': X_val,
		'y_val': y_val,
		'mu': mu,
		's2': s2,
	}

def load_test_data(sorted_path, person):
	X = np.load(F'{sorted_path}/{person}_test_brain.npy').astype('float32')
	y = np.load(F'{sorted_path}/{person}_test_event.npy').astype('int')

	X, mu, s2 = normalization(X)
	X = X.transpose(0, 1, 3, 2)
	return X, y