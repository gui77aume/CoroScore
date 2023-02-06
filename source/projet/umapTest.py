import matplotlib.pyplot as plt
from umap import UMAP

import donnees3PROD as d

X = d.X_complet
y = d.y_complet

for k in [60, ]:
    for min_distance in [ .4]:
        fig = plt.figure()
        ax = fig.add_subplot()
        um = UMAP(n_components=2, n_neighbors=k, min_dist=min_distance)
        embeddings = um.fit_transform(X)
        ax.scatter(embeddings[:, 0], embeddings[:, 1], c=y)
        plt.title(f"UMAP (n_neighbors={k} mdist={min_distance})")
        plt.show()
