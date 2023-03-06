# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 17:40:27 2023

@author: shani
"""

import numpy as np
import pandas as pd
import xlrd
from matplotlib.pyplot import figure, plot, title, legend, xlabel, ylabel, show
import matplotlib.pyplot as plt
from scipy.linalg import svd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 

filename = "C:/Users/shani/OneDrive - Danmarks Tekniske Universitet/5. semester/02450 introduktion til machine learning og data mining/GlassIdentification/Data/glass.data"
df = pd.read_csv(filename)

stats = df.describe()

raw_data = df.values
columns = range(0,11)
x = raw_data[:,columns]

attributeNames = np.asarray(df.columns[columns])
# %%
### data visualization
dd = pd.DataFrame(x[:,9])
plt.figure()

# std data
features = [" RI", " Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]
xx = df.loc[:, features].values
xx = StandardScaler().fit_transform(xx)

#Check for missing values
df.isnull().sum()

#Check data types
df.dtypes

#Convert Type to categorical variable
df['Type'] = pd.Categorical(df['Type'])

sns.boxplot(x='Type',y='RI', data=df)
plt.show()

sns.pairplot(df)
plt.show()

sns.set(font_scale=1.2)
sns.set_style("whitegrid")
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), cmap="YlGnBu", annot=True)
plt.title("Correlation Heatmap for Glass Indentification Dataset")
plt.show()
# %%
# PCA
pca = PCA()
principleComponent = pca.fit_transform(xx)
pcaExplainedVariance = pca.explained_variance_ratio_
loadings = pca.components_

plt.scatter(principleComponent[:,0], principleComponent[:,1])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Glass Identification Dataset")
plt.show()

# %%
# Plot the principal directions
plt.figure(figsize=(8, 6))
plt.title("Principal Directions of the First Two PCA Components")
plt.xlabel("Loading for PC1")
plt.ylabel("Loading for PC2")
for i, feature in enumerate(features):
    plt.arrow(0, 0, loadings[0,i], loadings[1,i], color='k', alpha=0.8)
    plt.text(loadings[0,i]*1.15, loadings[1,i]*1.15, feature, color='k', ha='center', va='center')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.grid()
plt.show()
#%%
#Plot the explained variance ratios
plt.bar(range(1, len(pcaExplainedVariance)+1), pcaExplainedVariance)

plt.xlabel("Principal Component")
plt.ylabel("Proportion of Variance Explained")
plt.title("Explained Variance Ratios by Principal Component")
plt.show()
# %%
# figure over temp
fig = plt.figure()
fig.suptitle('Fe', fontsize=14, fontweight='bold')

ax = fig.add_subplot(111)
ax.boxplot(dd)
ax.grid()

ax.set_ylabel('Class')
plt.show()

# %%
#hist over temp
plt.hist(dd)
plt.title('Fe')
plt.xlabel('Class')
plt.ylabel('Obervations')
plt.grid(axis = 'x')
plt.show()

#%% PCA

classLabels = raw_data[:,-1]

classNames = np.unique(classLabels)

classDict = dict(zip(classNames, range(len(classNames))))

y = np.array([classDict[cl] for cl in classLabels])

N, M = x.shape

C = len(classNames)
#PCA 
#From ex2_1_1
# Extract vector y, convert to NumPy array
y = np.asarray([classDict[value] for value in classLabels])

# Preallocate memory, then extract csv data to matrix X
X = np.empty((214, 10))
for i, col_id in enumerate(range(0, 10)):
    X[:, i] = np.asarray(raw_data[:,col_id])

# Compute values of N, M and C.
N = len(y)
M = len(attributeNames)
C = len(classNames)

# Subtract mean value from data
Y = X - np.ones((N,1))*X.mean(axis=0)
# removes the standard deviation.
Y = Y/(np.ones((N,1))*X.std(axis=0))


# PCA by computing SVD of Y
U,S,V = svd(Y,full_matrices=False)


# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 


threshold = 0.9

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()

#end of PCA analisys
#PCA 
#From ex2_1_1
# Extract vector y, convert to NumPy array
y = np.asarray([classDict[value] for value in classLabels])

# Preallocate memory, then extract csv data to matrix X
X = np.empty((214, 10))
for i, col_id in enumerate(range(1, 10)):
    X[:, i] = np.asarray(raw_data[:,col_id])

# Compute values of N, M and C.
N = len(y)
M = len(attributeNames)
C = len(classNames)

# Subtract mean value from data
Y = X - np.ones((N,1))*X.mean(axis=0)
# removes the standard deviation.
Y = Y/(np.ones((N,1))*X.std(axis=0))


# PCA by computing SVD of Y
U,S,V = svd(Y,full_matrices=False)


# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 


threshold = 0.9

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()

#%% Doesnt work
# percent of the variance. Let's look at their coefficients:
U,S,Vh = svd(Y,full_matrices=False)
V=Vh.T
N,M = X.shape

pcs = [0,1,2]
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r','g','b']
bw = .2
r = np.arange(1,M+1)
for i in pcs:    
    plt.bar(r+i*bw, V[:,i], width=bw)
plt.xticks(r+bw, attributeNames)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('PCA Component Coefficients')
plt.show()