import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
from tensorboardX import SummaryWriter
from sklearn.model_selection import KFold
from torch import optim
from utils import data_path


class Solver:
    def __init__(self, model, data, vebose=True):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()  # nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=0.00005)
        # self.optlver import Solver
        self.writer = SummaryWriter()

        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']

        self.vebose = vebose

    def check_accuracy(self, X, y):
        ''' Evaluate function returns values of different criteria like accuracy, precision etc.
        '''
        results = []
        batch_size = 100
        predicted = []
        
        for i in range(len(X)//batch_size):
            s = i*batch_size
            e = i*batch_size+batch_size
            inputs = Variable(torch.from_numpy(X[s:e]).cuda(0))
            pred = self.model(inputs)
            predicted.append(pred.data.cpu().numpy())
        predicted = np.array(predicted).reshape(-1, 2)

        inputs = torch.from_numpy(X[len(X)//batch_size * batch_size:]).cuda(0)
        if inputs.shape[0] > 0:
            pred = self.model(inputs)
            predicted = np.concatenate((predicted, pred.data.cpu().numpy()))

        y_pred = np.argmax(predicted, axis=1)
        acc = np.mean(y_pred == y)
        return acc

    def train(self, batch_size=512, save_name='EEGNet.pth'):

        for epoch in range(3000):

            running_loss = 0.0
            for i in range(len(self.X_train)//batch_size-1):
                s = i*batch_size
                e = i*batch_size+batch_size
                inputs = torch.from_numpy(self.X_train[s:e])
                labels = torch.LongTensor(self.y_train[s:e])
                inputs, labels = Variable(inputs.cuda(0)), Variable(labels.cuda(0))

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.data
            
            # Validation accuracy
            train_acc = self.check_accuracy(self.X_train, self.y_train)
            val_acc = self.check_accuracy(self.X_val, self.y_val)
            if epoch % 50 == 0:
                print("\nEpoch ", epoch)
                print("Training Loss ", running_loss)
                print(F'Accuray: train({train_acc}), val({val_acc})')

            self.writer.add_scalar('p300/loss', running_loss, epoch)
            self.writer.add_scalar('p300/train_acc', train_acc, epoch)
            self.writer.add_scalar('p300/val_acc', val_acc, epoch)

            torch.save(self.model.state_dict(), F'{data_path}/{save_name}')


