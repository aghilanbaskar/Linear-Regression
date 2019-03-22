import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

#load datset
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

x_train = df_train['x']
y_train = df_train['y']
x_test = df_test['x']
y_test = df_test['y']

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

x_train = x_train.reshape(-1,1)
y_train= y_train.reshape(-1,1)

#find a and b
n=len(y_train)
l_r=0.0001 #learning rate
a = np.zeros((n,1))
b = np.zeros((n,1))

epochs = 0
while(epochs < 1000):
    y = a + b * x_train
    error = y - y_train
    mean_sq_er = np.sum(error**2)
    mean_sq_er = mean_sq_er/n
    a = a - l_r * np.sum(error)/n
    b = b - l_r * np.sum(error * x_train)/n
    epochs += 1

a=np.sum(a)/n
b=np.sum(b)/n

#predicting output for test data
y_pred=[]
for i in range(len(y_test)):
    y_predict=a+(b*x_test[i])
    y_pred.append(y_predict)
    print('y:',y_test[i],'predicted_y:',y_predict)

#accuracy
print('final_accuracy:',r2_score(y_test,y_pred))


#plotting the actual and predicted output
y_plot = []
for i in range(100):
    y_plot.append(a + b * i)
plt.figure(figsize=(10,10))
plt.scatter(x_test,y_test,color='red',label='GT')
plt.plot(range(len(y_plot)),y_plot,color='black',label = 'pred')
plt.legend()
plt.show()
