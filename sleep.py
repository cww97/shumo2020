import numpy as np
import pandas as pd
from input_data import shuffle_data
from sklearn.model_selection import train_test_split
from sklearn import svm


def load_data():
    excel_data = pd.read_excel('data/sleep.xlsx', sheet_name=[0, 1, 2, 3, 4])
    
    sleeps, labels = [], []
    for i in range(len(excel_data)):
        sleep = excel_data[i]
        sleep = np.array(sleep)[:, 1: 5].astype('float')
        label = np.full_like(sleep[:, 0], i).astype('int')
        
        sleeps.append(sleep)
        labels.append(label)
    
    sleeps = np.concatenate(sleeps)
    labels = np.concatenate(labels)
    # shuffle_data(sleeps, labels)

    mu = np.mean(sleeps, axis=0)
    s2 = np.std(sleeps, axis=0)
    sleeps = (sleeps - mu) / s2

    return train_test_split(sleeps, labels, test_size=0.1, random_state=2333)


def rua(X_train, X_test, y_train, y_test):
    clf = svm.SVC(kernel='precomputed')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = np.mean(y_pred == y_test)
    print(acc)


def huo(X_train, X_test, y_train, y_test):
    pass

if __name__ == "__main__":
    rua(*load_data())