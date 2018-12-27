import citiapi
import pandas as pd
import datetime as dt
import math
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.cm as cm

varswap = pd.read_csv('varswap.csv')
level = pd.read_csv('level.csv')

'''
def calculate_realized(series):
    data = np.asarray(series)
    numerator = data.copy()
    numerator = np.delete(numerator, numerator.__len__() - 1)
    denominator = data.copy()[1:data.__len__()]

    return math.sqrt(np.sum(np.square(np.log(numerator / denominator))) * (252 / numerator.__len__())) * 100

pivot = level.pivot(index='date', columns='name',values='value')
pivot = pivot.fillna(method='ffill')
pivot.sort_index(ascending=False,inplace=True)
three = dt.datetime.strptime(str(max(pivot.index)), '%Y%m%d') - dt.timedelta(days=365)
start = three.year*10000+three.month*100+three.day

realized = pivot[pivot.index > start]

for items in list(realized.index):
    for items1 in list(realized.columns):
        before = dt.datetime.strptime(str(items), '%Y%m%d') - dt.timedelta(days=30)
        begin = before.year * 10000 + before.month * 100 + before.day
        temp = pivot[pivot.index <= items]
        realized.loc[items,items1] = calculate_realized(temp[temp.index>begin].loc[:,items1])

realized.columns = ['Realized_SPX', 'Realized_SX5E']

varswap_1m = varswap.pivot(index='date', columns='name',values='value')
varswap_1m = varswap_1m.fillna(method='ffill')
varswap_1m.sort_index(ascending=False,inplace=True)
start = three.year*10000+three.month*100+three.day
varswap_1m = varswap_1m[varswap_1m.index > start]
varswap_1m.to_csv('varswap_1m.csv')
'''
varswap_1m = pd.read_csv('varswap_1m.csv')
varswap_1m = varswap_1m*100
varswap_1m['date'] = varswap_1m['date']//100
realized = pd.read_csv('realized.csv')
realized = realized.set_index('date')
varswap_1m = varswap_1m.set_index('date')
plt.plot(list(range(0,realized.__len__(),1)),realized['Realized_SPX'][::-1],c='red')
plt.plot(list(range(0,realized.__len__(),1)),realized['Realized_SX5E'][::-1],c='blue')
plt.show()

average_realized = pd.DataFrame()
average_varswap = pd.DataFrame()

temp = realized.copy()
temp1 = varswap_1m.copy()
for i in range(realized.__len__()-49):
    average_realized = average_realized.append(pd.DataFrame(temp[:50].mean()).T)
    temp = temp.shift(-1)

    average_varswap = average_varswap.append(pd.DataFrame(temp1[:50].mean()).T)
    temp1 = temp1.shift(-1)

average_realized.index=realized.index.values[:-49]
average_varswap.index=varswap_1m.index.values[:-49]

average_spread = average_varswap['SPX']-average_varswap['STOXX50E']
realized_spread = realized['Realized_SPX']-realized['Realized_SX5E']
var_spread = varswap_1m['SPX']-varswap_1m['STOXX50E']
plt.scatter(realized_spread[realized_spread.index >= min(var_spread.index)].values,var_spread.values,c='red')
plt.xlabel('realized spread')
plt.ylabel('var spread')
plt.show()



merged =pd.concat([realized_spread[realized_spread.index >= min(average_spread.index)], average_spread], axis=1)
X_std = StandardScaler().fit_transform(merged)
sklearn_pca = PCA(n_components=2)
Y_sklearn = sklearn_pca.fit_transform(X_std)

Y_sklearn = pd.DataFrame(Y_sklearn)
Y_sklearn.index = merged.index
final_df = pd.concat([Y_sklearn,var_spread[var_spread.index>=min(merged.index)]], axis=1)
final_df.columns = ['pca1', 'pca2','spread']
step_size = 2.5
step1 =  [item//step_size for item in list(final_df['spread'].values)]

unique_steps = set(step1)
colors = ['blue', 'green', 'red', 'cyan', 'magenta' ,'yellow', 'black', 'white']
unique_steps = list(unique_steps)
unique_steps.sort()
count = 0
for item in unique_steps:
    if count == 0:
        temp = final_df[final_df['spread']<unique_steps[count]]
    elif count == unique_steps.__len__()-1:
        temp = final_df[final_df['spread'] > unique_steps[count]]
    else:
        temp = final_df[final_df['spread'] > unique_steps[count]]
        temp = temp[temp['spread'] > unique_steps[count+1]]
    plt.scatter(temp['pca1'],temp['pca2'],c=colors[count])
    count = count+1
plt.xlabel('pca1')
plt.ylabel('pca2')

plt.show()
