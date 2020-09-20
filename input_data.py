from openpyxl import load_workbook
import numpy as np
import pandas as pd
from utils import data_path
from utils import get_p300_num
import cv2


sorted_path = F'{data_path}/sorted_data'
persons = ['S1', 'S2', 'S3', 'S4', 'S5']
seg_num = 60
siz = (46, 20)


def _deal_events(xl):
	xl = xl.parse(xl.sheet_names, header=None)
	labels, events, indexs = [], [], []
	for sheet in xl.keys():
		event = np.array(xl[sheet])[:, 0]
		index = np.array(xl[sheet])[:, 1]

		row, col = get_p300_num(sheet[-2])
		label = np.full_like(event, 0)
		label[event==row] = 1
		label[event==col] = 1
		
		index = index[event <= 12]
		label = label[event <= 12]
		event = event[event <= 12]

		events.append(event)
		labels.append(label)
		indexs.append(index)

	events = np.concatenate(events)
	labels = np.concatenate(labels)
	indexs = np.concatenate(indexs)
	return events, labels, indexs

def _deal_brain(xl, indexs):
	xl = xl.parse(xl.sheet_names, header=None)
	brains = []
	cnt = 0
	for sheet in xl.keys():
		index = indexs[cnt*60: (cnt+1)*60]
		brain = np.array(xl[sheet])

		waves = []
		wave = brain[np.newaxis, index[0] - 44: index[0]]
		wave = np.resize(wave, (40, 20))
		waves.append(wave[np.newaxis, :, :])
		for i in range(1, len(index)):
			wave = brain[index[i-1]: index[i]]
			wave = np.resize(wave, (40, 20))
			waves.append(wave[np.newaxis, :, :])
		waves = np.concatenate(waves)[:, np.newaxis, :, :]

		brains.append(waves)
		cnt += 1
	brains = np.concatenate(brains)

	return brains

def load_one_person_data(person, mode='train'):
	print(person, mode)
	this_path = F'data/{person}'
	xl = pd.ExcelFile(F'{this_path}/{person}_{mode}_event.xlsx')
	events, labels, indexs = _deal_events(xl)

	xl = pd.ExcelFile(F'{this_path}/{person}_{mode}_data.xlsx')
	brains = _deal_brain(xl, indexs)

	output_path = sorted_path
	np.save(F'{output_path}/{person}_{mode}_brain', brains)
	np.save(F'{output_path}/{person}_{mode}_event', events)
	if mode == 'train':
		np.save(F'{output_path}/{person}_{mode}_label', labels)
	return output_path

def read_write_data():
    for person in persons:
        load_one_person_data(person, 'train')
        load_one_person_data(person, 'test')
#----------------------------------------------------------------------------

def balance_data(X, y):
	# balance positive/negative data
	positive_X = X[y==1].repeat(4, 0)
	positive_y = y[y==1].repeat(4, 0)
	X = np.concatenate((X, positive_X))
	y = np.concatenate((y, positive_y))
	return X, y

def shuffle_data(X, y):
	# shuffle
	idx = np.arange(X.shape[0])
	np.random.shuffle(idx)
	return X[idx], y[idx]

def rua(X):
	rua = []
	for i in range(X.shape[0]):
		x = X[i, 0]
		x = cv2.resize(x, (32, 32), interpolation=cv2.INTER_CUBIC)
		rua.append(x[np.newaxis, np.newaxis,:, :])
	rua = np.concatenate(rua)
	return rua

def sort_train_data(sorted_path=sorted_path, persons=persons):
	# print(sorted_path)
	X, y = [], []
	selected_persons = persons
	print(selected_persons)
	for person in selected_persons:
		X.append(np.load(F'{sorted_path}/{person}_train_brain.npy'))
		y.append(np.load(F'{sorted_path}/{person}_train_label.npy'))
	X = np.concatenate(X)
	y = np.concatenate(y)

	X, y = balance_data(X, y)
	X, y = shuffle_data(X, y)
	return X, y


if __name__ == '__main__':
    # read_write_data()
    sort_train_data()
