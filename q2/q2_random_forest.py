# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 13:05:50 2020

@author: Fengyu Han
"""
import os 
import numpy as np
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
files= os.listdir('C:\\Users\\lenovo\\Desktop\\建模\\2020年中国研究生数学建模竞赛赛题\\sorteddata')
# X_data = np.load('C:\\Users\\lenovo\\Desktop\\建模\\2020年中国研究生数学建模竞赛赛题\\X.npy')
# Y_data = np.load('C:\\Users\\lenovo\\Desktop\\建模\\2020年中国研究生数学建模竞赛赛题\\Y.npy')
X_train=[]
Y_train=[]
X_test=[]
Y_test=[]
loadpath='C:\\Users\\lenovo\\Desktop\\建模\\2020年中国研究生数学建模竞赛赛题\\sorteddata\\'
for file in files:
    if 'train' in file :
        if 'label' in file:
            Y_train.append(np.load(loadpath+file))
        elif 'brain' in file:
            X_train.append(np.load(loadpath+file))
    elif 'test' in file:
        if 'label' in file:
            Y_test.append(np.load(loadpath+file))
        elif 'brain' in file:
            X_test.append(np.load(loadpath+file))
X_train = np.concatenate(X_train)
Y_train = np.concatenate(Y_train)
X_test = np.concatenate(X_test)
merge_X=np.mean(X_train,axis=2)
X_test= np.mean(X_test,axis=2)
X_test= np.squeeze(X_test)
X=np.squeeze(merge_X,axis=1)
X_data=pd.DataFrame(X,columns=['C'+str(i) for i in range(1,21)])
clf = RandomForestClassifier(n_estimators=100,max_depth=5,class_weight={0:1,1:5},criterion='entropy')
clf.fit(X_data,Y_train)
clf.predict(X_test)
importances = clf.feature_importances_


#重要程度
plt.figure(figsize=(10,6))
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.title('通道的重要程度排序',fontsize=20)
plt.ylabel("import level",fontsize = 15,rotation=90)
sort_importance=np.argsort(importances)[::-1]
for i in sort_importance:
    plt.bar(X_data.columns[i],importances[i],color='orange',align='center')
    plt.xticks(np.arange(20),X_data.columns[sort_importance],rotation=90,fontsize=15)
plt.show()

import os
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import auc,roc_curve
import matplotlib.pyplot as plt
 
from sklearn.linear_model import LogisticRegression

model=LogisticRegression(penalty='l1',C=1,class_weight='balanced',solver='liblinear')
model.fit(X_data,Y_train)
model.predict_proba(X_data)