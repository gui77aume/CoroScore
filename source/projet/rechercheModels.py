import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn_genetic.space import Categorical, Continuous, Integer

import Optim
import donnees3PROD as d
import evalModelFonction

# Script de sélection de famille d'algorithme
# Pour chaque algo, effectue une selection génétique des hyperparametres
# et affiche les performances (accuracy, ROC AUC, sensibilité et spécificité VS seuil)
# Les données doivent être paramétrées dans donnes3PROD et UI

runsvc = False  # tous les noyaux
runsvc2 = True  # noyaux linéaire seulement

runsvclin = False
rundt = True
runadaboost = False
runmlp = True

if runsvc:
    param_grid_svc = {
        'C': Continuous(0.01, 100),
        'gamma': Continuous(0.01, 10),
        'degree': Integer(2, 3),
        'kernel': Categorical(['linear', 'rbf', 'sigmoid', 'poly']),
        'tol': Continuous(1e-4, 1e-2)
    }
    opt = Optim.Optim(svm.SVC(class_weight='balanced', probability=True), param_grid_svc)
    opt_mod = opt.run(d.X_train, d.y_train)
    # plot_fitness_evolution(opt_mod)
    mod2 = opt_mod.best_estimator_
    b = opt_mod.best_params_
    nom = f"SVC avec {b}"
    evalModelFonction.eval_modele(mod2, d.X_train, d.y_train, d.X_test, d.y_test, nomModele=nom)
    print(f"\nMeilleurs parametres SVC : {opt_mod.best_params_}\n\n")
    # fin SVC

if runsvc2:
    param_grid_svc = {
        'C': Continuous(0.01, 100),
        'tol': Continuous(1e-4, 1e-2)
    }
    opt = Optim.Optim(svm.SVC(class_weight='balanced', probability=True, kernel='linear'), param_grid_svc)
    opt_mod = opt.run(d.X_train, d.y_train)
    # plot_fitness_evolution(opt_mod)
    mod2 = opt_mod.best_estimator_
    b = opt_mod.best_params_
    nom = f"SVC avec {b}"
    evalModelFonction.eval_modele(mod2, d.X_train, d.y_train, d.X_test, d.y_test, nomModele=nom)
    print(f"\nMeilleurs parametres SVC : {opt_mod.best_params_}\n\n")
    # fin SVC

if rundt:
    param_grid_dt = {
        'criterion': Categorical(["gini", "log_loss", "entropy"]),
        'splitter': Categorical(["best", "random"]),
        'ccp_alpha': Continuous(0.001, 0.5)

    }
    optDT = Optim.Optim(DecisionTreeClassifier(class_weight='balanced'), param_grid_dt)
    modDT = optDT.run(d.X_train, d.y_train)
    # plot_fitness_evolution(modDT)
    bestDT = modDT.best_estimator_
    evalModelFonction.eval_modele(bestDT, d.X_train, d.y_train, d.X_test, d.y_test, nomModele="Arbre de décision")
    # TODO : afficher DT entraineé sur tout
    print(f"\nmeilleurs paramètres DT : {modDT.best_params_}\n")

    # fin DT

if runadaboost:
    param_grid_ada = {
        'n_estimators': Integer(200, 500),
        'learning_rate': Continuous(0.1, 3),
        'base_estimator__max_depth': Integer(5, 15)
    }
    optAda = Optim.Optim(
        AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='log_loss', splitter='random',
                                                                 ccp_alpha=0.032)), param_grid_ada)

    ogm_ada = optAda.run(d.X_train, d.y_train)
    # plot_fitness_evolution(ogm_ada)
    evalModelFonction.eval_modele(ogm_ada.best_estimator_, d.X_train, d.y_train, d.X_test, d.y_test,
                                  nomModele="ADABOOST")
    print(f"\nMeilleurs paramètres ADABOOST : {ogm_ada.best_params_}\n")

    # fin adaboost

if runmlp:
    mlp_h_layers = (20, 20,)
    param_grid_mlp = {
        'activation': Categorical(['relu', 'logistic', 'tanh']),
        'max_iter': Integer(200, 2000),
        'beta_1': Continuous(0.1, 0.99),  # adam uniqt
        'beta_2': Continuous(0.1, 0.99999),  # adam uniqt
        'epsilon': Continuous(1e-9, 1e-7),  # adam uniqt
        'n_iter_no_change': Integer(5, 50),
        'power_t': Continuous(0.1, 1),  # sgd et invscaling
        'validation_fraction': Continuous(0.1, 0.5),  # early stopping
        'tol': Continuous(1e-5, 1e-1, distribution='log-uniform'),
        'alpha': Continuous(1e-5, 10),  # terme de régularisation L2
        'learning_rate': Categorical(['constant', 'invscaling']),
        'learning_rate_init': Continuous(1e-4, 0.3),
        'momentum': Continuous(0.1, 0.9),
        'solver': Categorical(['sgd', 'adam']),
        'batch_size': Integer(5, 90)
    }
    op2 = Optim.Optim(MLPClassifier(hidden_layer_sizes=mlp_h_layers), param_grid_mlp)
    op2_mod = op2.run(d.X_train, d.y_train)
    # plot_fitness_evolution(op2_mod)
    nom = f"MLP"
    evalModelFonction.eval_modele(op2_mod.best_estimator_, d.X_train, d.y_train, d.X_test, d.y_test, nomModele='MLP')
    print(f"\nMeilleurs parametres MLP avec archi {mlp_h_layers} : {op2_mod.best_params_} \n\n")

    # fin MLP

plt.show()
