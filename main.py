import numpy as np
import torch
from models import EEGNet
from solver import Solver
from data_loader import load_data, load_test_data
from torch.autograd import Variable
from p300 import P300
from utils import data_path
from input_data import sorted_path, persons
from models.LeNet import LeNet5
import sys


# model = LeNet5().cuda(0)
model = EEGNet().cuda(0)
selected_id = 4
selected_persons = persons[selected_id: selected_id+1]
model_name = F'model_{persons[selected_id]}.pth'
# model_name = 'best_EEGNet.pth'


def train():
    data = load_data(persons=selected_persons)
    solver = Solver(model=model, data=data)
    solver.train(save_name=model_name)


def q1():
    '''Question 1
    '''
    model.load_state_dict(torch.load(F'{data_path}/{model_name}'))
    model.eval()
    p300 = P300(model=model)

    for person in selected_persons:
        brain, event = load_test_data(sorted_path, person)
        target = p300.get_target(brain, event)
        print(target)


if __name__ == "__main__":
    exec(sys.argv[1] + '()')
