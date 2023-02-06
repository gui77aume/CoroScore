import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn_genetic import GASearchCV


class Optim:
    """Pratique une optimisation de modele d'une famille donnée (ex SVC)
     avec algo genetique et fournit :
     -les hyperparamètres du meilleur modèle et le meilleur modele"""
    _mod = None
    _param_grid = None
    _n = 3
    _evolved_mod = None

    def __init__(self, model, param_grid, n_folds=5):
        self.set_modele(model, param_grid, n_folds)

    def set_modele(self, model, param_grid, n_folds=5):
        self._mod = model
        self._param_grid = param_grid
        self._n = n_folds
        """Definition du modèle dont on cherche les hyperparametres"""
        pass

    def run(self, X:np.array, y: np.array):
        """Lance la selection génétique"""

        cv = StratifiedKFold(n_splits=self._n, shuffle=True)

        self._evolved_mod = GASearchCV(estimator=self._mod,
                                       cv=cv,
                                       scoring='accuracy',
                                       param_grid=self._param_grid,
                                       n_jobs=-1,
                                       verbose=True,
                                       generations=80
                                       )

        self._evolved_mod.fit(X, y)
        return self._evolved_mod

    def get_ev_model(self):
        """renvoie le modele obtenu"""
        return self._evolved_mod
        pass


