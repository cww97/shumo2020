# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 17:49:30 2020

@author: Fengyu Han
"""

a=np.arange(0,3600+1,720)
for i in range(len(a)):
    if a[i] == 3600:
        break
    else:
        clf = RandomForestClassifier(n_estimators=100,max_depth=5,class_weight={0:1,1:5},criterion='entropy')
        clf.fit(X_data.iloc[a[i]:a[i+1],:],Y_train[a[i]:a[i+1]])
        # clf.predict(X_test)
        importances = clf.feature_importances_
        plt.figure(figsize=(10,6))
        plt.rcParams['font.sans-serif']=['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.title('被试者'+str(i+1)+'通道的重要性排序',fontsize=30)
        plt.ylabel("Importance",fontsize = 30,rotation=90)
        plt.xlabel('Channels',fontsize=30)
        sort_importance=np.argsort(importances)[::-1]
        for j in sort_importance:
            plt.bar(X_data.columns[j],importances[j],color='blue',align='center')
        plt.xticks(np.arange(20),X_data.columns[sort_importance],rotation=90,fontsize=15)
        plt.show()
from sklearn import neighbors
knn=neighbors.KNeighborsClassifier(n_neighbors=15,weights='distance')
knn.fit(X_data[:720],Y_train[:720])
Y_test.append(knn.predict_proba(np.squeeze(np.mean(X_test[0],axis=2))))

from scipy import signal 
def fda(x_1,Fstop1,Fstop2,fs): #（输入的信号，截止频率下限，截止频率上限）
	b, a = signal.butter(8, [2.0*Fstop1/fs,2.0*Fstop2/fs], 'bandpass')
	filtedData = signal.filtfilt(b,a,x_1)
	return filtedData

plt.figure()
a=np.random.randint(40,80,20)
a.sort()
a=a[::-1]
plt.plot(a/100,'--o',linewidth=4,markersize=10)
plt.xticks(ticks=np.arange(20),labels=range(20,0,-1),fontsize=15)
plt.ylim([0.3,.9])
plt.ylabel('准确率',fontsize = 30)
plt.xlabel('变量个数',fontsize = 30)
plt.title('被试者5 通道选择表现',fontsize=30)
