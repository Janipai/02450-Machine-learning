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
dd = pd.DataFrame(x[:,10])
plt.figure()

# std data
features = [" RI", " Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]
xx = df.loc[:, features].values
xx = StandardScaler().fit_transform(xx)

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
fig.suptitle('Type of glass', fontsize=14, fontweight='bold')

ax = fig.add_subplot(111)
ax.boxplot(dd)
ax.grid()

ax.set_ylabel('Class')
plt.show()

# %%
#hist over temp
plt.hist(dd)
plt.title('Type of glass')
plt.xlabel('Class')
plt.ylabel('Obervations')
plt.grid(axis = 'x')
plt.show()

# %%
# 2.1.1

X = np.empty((90, 8))
for i, col_id in enumerate(range(3, 11)):
    X[:, i] = np.asarray(df.col_values(col_id, 2, 92))

# %%
# 2.1.3
# Subtract mean value from data
Y = X - np.ones((N,1))*X.mean(axis=0)

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