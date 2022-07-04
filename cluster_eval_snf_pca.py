from sklearn.preprocessing import minmax_scale,scale,StandardScaler
import numpy as np
import os
from SNF import *
#载入数据
folder_CCLE='./data/Datasets/CCLE_dataset/dataset'
folder_GDSC='./data/Datasets/GDSC_dataset/dataset'
def load_cell_lines(folder):             
    Nets = []
    input_dims = []
    with open(os.path.join(folder, "Similarity matrix based on gene expression profile.txt"), "r") as raw:
        #raw.next()
        cell_sim_1 = np.array([line.strip("\n").split()[0:] for line in raw])
        
    with open(os.path.join(folder, "Similarity matrix based on copy number alteration.txt"), "r") as raw:
        #raw.next()
        cell_sim_2 = np.array([line.strip("\n").split()[0:] for line in raw])
        
    with open(os.path.join(folder, "Similarity matrix based on single nucloetid mutation.txt"), "r") as raw:
        #raw.next()
        cell_sim_3 = np.array([line.strip("\n").split()[0:] for line in raw])
    Nets.extend([minmax_scale(cell_sim_1),minmax_scale(cell_sim_2),minmax_scale(cell_sim_3)])
    input_dims = [cell_sim_1.shape[1],cell_sim_2.shape[1], cell_sim_3.shape[1]]
    return Nets, input_dims

Nets_cellline, input_dims_cellline = load_cell_lines(folder_GDSC)#folder_GDSC

#print(Nets_cellline, input_dims_cellline)
import copy 
Nets_cellline2 = Nets_cellline.copy()
#SNF融合
snf_fusion=SNF(Nets_cellline2,5,10)
snf_fusion = minmax_scale(snf_fusion)
mda = np.loadtxt("celllinemdaFeatures.txt")#celllinemdaFeatures
#t-sne降维
from sklearn.manifold import TSNE
X_embedded = TSNE().fit_transform(mda)
# print(X_embedded.shape)


# from sklearn.decomposition import PCA
# X_embedded = PCA(n_components=10).fit_transform(snf_fusion)

print("x_embeded shape:",X_embedded.shape)

#mean-shift聚类
from sklearn.cluster import MeanShift,estimate_bandwidth
bandwidth = estimate_bandwidth(X_embedded)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X_embedded)
labels = ms.labels_
cluster_centers = ms.cluster_centers_
labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)
print('bandwidth:%f'%bandwidth)
print("number of estimated clusters : %d" % n_clusters_)


#DBSCAN 聚类

# from sklearn.cluster import DBSCAN
# clustering = DBSCAN(eps=0.3, min_samples=10).fit(X_embedded)
# labels = clustering.labels_
# labels_unique = np.unique(labels)
# n_clusters_ = len(labels_unique)
# print("number of estimated clusters : %d" % n_clusters_)

#聚类评价
from sklearn import metrics
print("Silhouette Coefficient(representing the coherence of clusters): %0.3f"
      % metrics.silhouette_score(X_embedded, labels,metric='sqeuclidean'))


# Plot result
#==============================================
import matplotlib.pyplot as plt
from itertools import cycle

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
index = np.array([])
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    index = np.append(index, np.where(labels == k))
    cluster_center = cluster_centers[k]
    plt.plot(X_embedded[my_members, 0], X_embedded[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
plt.xlabel('tsne-2d-one', fontweight='bold') #
plt.ylabel('tsne-2d-two',fontweight='bold') #
plt.title('GDSC\'s Cell Lines Cluster based on MDA',fontweight='bold') #
plt.savefig('abc.png')
index = index.flatten().astype(int)

#=================================================================================
from sklearn.metrics.pairwise import euclidean_distances
fused_sim=minmax_scale(-euclidean_distances(X_embedded[index],X_embedded[index]))

# Plot heatmap

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


plt.clf()
ax = sns.heatmap(Nets_cellline[0], fmt="d",cmap='YlGnBu')
plt.xlabel('cell lines in GDSC', fontweight='bold') #
plt.ylabel('cell lines in GDSC',fontweight='bold') #
plt.title('Similarity based on gene expression',fontweight='bold') #
plt.savefig('a.png')
plt.clf()
ax = sns.heatmap(Nets_cellline[1], fmt="d",cmap='YlGnBu')
plt.xlabel('cell lines in GDSC', fontweight='bold') #
plt.ylabel('cell lines in GDSC',fontweight='bold') #
plt.title('Similarity based on copy number alteration',fontweight='bold') #
plt.savefig('b.png')
plt.clf()
ax = sns.heatmap(Nets_cellline[2], fmt="d",cmap='YlGnBu')
plt.xlabel('cell lines in GDSC', fontweight='bold') #
plt.ylabel('cell lines in GDSC',fontweight='bold') #
plt.title('Similarity based on single nucloetid mutation',fontweight='bold') #
plt.savefig('c.png')
plt.clf()
ax = sns.heatmap(fused_sim, fmt="d",cmap='YlGnBu')
plt.xlabel('cell lines in GDSC', fontweight='bold') #
plt.ylabel('cell lines in GDSC',fontweight='bold') #
plt.title('Similarity based on MDA',fontweight='bold') #

plt.savefig('aaa.png')



import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame
data={'cell-line':['GDSC','GDSC','CCLE','CCLE'],
      'method':['SNF','MDA','SNF','MDA'],
      'silhouette_score':[0.609,0.794,0.642,0.770]}
df = DataFrame(data)
sns.set_context('paper')
sns.barplot(y='silhouette_score',x='cell-line',data=df,hue='method')
plt.savefig('ae.png')