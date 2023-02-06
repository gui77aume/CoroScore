import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import auc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

# Fonction d'aide à l'évaluation d'un modèle
# Affiche les courbes ROC et l'AUC sur trois sous-ensembles des données d'apprentissage tirés aléatoirement
# ainsi que sur les données de test

# Affiche également le graphe spécificité et sensibilité VS seuil de proba

# basé sur :
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html


def eval_modele(classifier, X_apprentissage, y_apprentissage, X_validation, y_validation, n_splits=3, nomModele=""):
    cv = StratifiedKFold(n_splits=n_splits)
    tprs = []
    aucs = []
    accs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(figsize=(6, 6))

    for fold, (train, test) in enumerate(cv.split(X_apprentissage, y_apprentissage)):
        classifier.fit(X_apprentissage[train], y_apprentissage[train])
        accs.append(classifier.score(X_apprentissage[test], y_apprentissage[test]))
        viz = RocCurveDisplay.from_estimator(
            classifier,
            X_apprentissage[test],
            y_apprentissage[test],
            name=f"ROC sur fold {fold + 1} (accy = {accs[-1]:.2f}), ",
            ax=ax,
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"ROC moyenne (AUC = %0.2f $\pm$ %0.2f, accy = %0.2f $\pm$ %0.2f )" % (
            mean_auc, std_auc, mean_acc, std_acc),
        lw=2,
        alpha=0.8,
    )

    # courbe de test sur données jamais vues
    # avec apprentissage sur toutes les données vues
    classifier.fit(X_apprentissage, y_apprentissage)
    acc_val = classifier.score(X_validation, y_validation)
    viz = RocCurveDisplay.from_estimator(
        classifier,
        X_validation,
        y_validation,
        name=f"ROC sur données de test (accy={acc_val:.2f})",
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)
    #

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 ecart type",
    )
    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        xlabel="1 - spécificité",
        ylabel="Sensibilité",
        title=nomModele
    )
    ax.axis("square")
    ax.legend(loc="lower right")


    # Graphe spécificité - sensibilité VS seuil proba
    proba = []
    sensibilite = []
    specificite = []
    for seuil in range(100):
        seuil /= 100
        cm = confusion_matrix(y_validation, classifier.predict_proba(X_validation)[:, 1] > seuil)
        tn, fp, fn, tp = cm.ravel()
        proba.append(seuil)
        sensibilite.append(tp / (tp + fn))
        specificite.append(tn / (tn + fp))

    figSSP = plt.figure()
    ax2SSP = figSSP.add_subplot(111)
    ax2SSP.set(
        xlim=[0., 1.05],
        ylim=[-0.05, 1.05],
        xlabel="Seuil de probabilité",
        ylabel="Sensibilité et spécificité",
        title=f"{nomModele}\nSensibilité et spécificité en fonction du seuil de probabilité"
    )
    ax2SSP.plot(proba, sensibilite,c='b', label="Sensibilité")
    ax2SSP.plot(proba, specificite,c='r', label="Spécificité")
    ax2SSP.legend(loc="lower left")
