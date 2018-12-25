import citiapi
import pandas as pd
import numpy as np
import datetime as dt
import math
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def calculate_realized(series):
    data = np.asarray(series)
    numerator = data.copy()
    numerator = np.delete(numerator, numerator.__len__() - 1)
    denominator = data.copy()[1:data.__len__()]

    return (math.sqrt(np.sum(np.square(np.log(numerator / denominator))) * (252 / numerator.__len__())) * 100)


level = pd.read_csv('index_level.csv')
pivot = level.pivot(index='date', columns='name',values='value')
pivot = pivot.fillna(method='ffill')
pivot.sort_index(ascending=False,inplace=True)

three = dt.datetime.strptime(str(max(pivot.index)), '%Y%m%d') - dt.timedelta(days=365)
start = three.year*10000+three.month*100+three.day

varswap = pd.read_csv('index_varswap.csv')
varswap = varswap.pivot(index = 'date', values='value',columns='name')
varswap = varswap[varswap.index > start]
varswap = varswap.fillna(method='ffill')
varswap.sort_index(ascending=False,inplace=True)
varswap = varswap*100

realized = pivot[pivot.index > start]

for items in list(realized.index):
    for items1 in list(realized.columns):
        before = dt.datetime.strptime(str(items), '%Y%m%d') - dt.timedelta(days=90)
        begin = before.year * 10000 + before.month * 100 + before.day
        temp = pivot[pivot.index <= items]
        realized.loc[items,items1] = calculate_realized(temp[temp.index>begin].loc[:,items1])

realized.columns = ['Realized_SPX', 'Realized_SX7E']
varswap.columns = ['varswap_SPX','varswap_SX7E']
spread = varswap.iloc[:,1]-varswap.iloc[:,0]
spread = spread - spread.mean()
spread = spread/spread.std()

merged = pd.concat([realized, varswap], axis=1)
merged = merged.fillna(method='ffill')


normalized = merged - merged.mean()
normalized = normalized/normalized.std()
normalized.columns = merged.columns
normalized.index = merged.index

'''
normalized = pd.DataFrame(StandardScaler().fit_transform(merged))
normalized.columns = merged.columns
normalized.index = merged.index
'''

correlation = pd.DataFrame(plt.matshow(normalized.corr())._A)
A = np.array(correlation)
# A = np.array(normalized.cov())
w, v = LA.eig(A)
percentage = [item/sum(w)for item in w]
first_component = max(percentage) # en yüksek eigenvalue

first_index = percentage.index(first_component) # en yüksek eigenvalue sahip index
first_index = 1
projection = v[0][first_index]*normalized.iloc[:,0]+v[1][first_index]*normalized.iloc[:,1]+v[2][first_index]*normalized.iloc[:,2]+v[3][first_index]*normalized.iloc[:,3]

final_df = pd.concat([projection, spread], axis=1)
#print(pd.DataFrame(plt.matshow(final_df.corr())._A))

print('percentage = ' + str(percentage[first_index]))
print('eigenvector = '+ str(v[:][first_index]))
print('correlation = '+ str(pd.DataFrame(plt.matshow(final_df.corr())._A).iloc[1,0]))
fit = np.polyfit(list(final_df[0]), list(final_df[1]), 1)

y_1 = fit[0] * min(list(final_df[0])) + fit[1]
y_2 = fit[0] * max(list(final_df[0])) + fit[1]

plt.scatter(list(final_df[0]), list(final_df[1]))
plt.plot([min(list(final_df[0])),max(list(final_df[0]))],[y_1,y_2],c='red')

plt.show()

''' alternative:  by  using sklearn '''

pca = PCA(n_components=4)
pca.fit(normalized.values)
print('eigen values = '+str(pca.explained_variance_ratio_))

transformed_data = pd.DataFrame(pca.transform(normalized.values))

transformed_data.index = normalized.index
final_df_1 = pd.concat([transformed_data, spread], axis=1)
correlation1 = pd.DataFrame(plt.matshow(final_df_1.corr())._A)
print('correlation ='+str(correlation1.iloc[:,-1]))
index_first = 1
fit = np.polyfit(list(final_df_1.iloc[:,index_first]), list(final_df_1.iloc[:,4]), 1)

y_1 = fit[0] * min(list(final_df_1.iloc[:,index_first])) + fit[1]
y_2 = fit[0] * max(list(final_df_1.iloc[:,index_first])) + fit[1]

plt.scatter(list(final_df_1.iloc[:,index_first]), list(final_df_1.iloc[:,4]))
plt.plot([min(list(final_df_1.iloc[:,index_first])),max(list(final_df_1.iloc[:,index_first]))],[y_1,y_2],c='red')

print(str(fit[0])+' '+ str(fit[1]))
plt.show()
