#  visualisation PCA

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

import donnees3PROD as d

noms = np.loadtxt('/home/guillaume/TPPyCharm/projet/heart.dat', dtype=np.str, delimiter=' ', usecols=14)

pcaCR = PCA()
pcaCR.fit(d.X_quantitatives)

ratios = pcaCR.explained_variance_ratio_
print(ratios)
plt.bar(range(len(ratios)), ratios)
plt.xticks(range(len(ratios)))
plt.xlabel("Composante principale")
plt.ylabel("% de variance expliquée")
plt.show()

mNt = pcaCR.transform(d.X_quantitatives)
print(mNt[:2, :])

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
for i in range(len(noms)):
    x, y = mNt[i, 0], mNt[i, 1]
    if d.y_complet[i] == 0:
        c = 'g'
    else:
        c = 'r'
    ax.scatter(x, y, c=c)
    ax.text(x, y, noms[i])

plt.show()

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
for i in range(len(noms)):
    x, y, z = mNt[i, 0], mNt[i, 1], mNt[i, 2]
    if d.y_complet[i] == 0:
        c = 'g'
    else:
        c = 'r'
    ax.scatter(x, y, z, c=c)
    ax.text(x, y, z, noms[i])

plt.show()
