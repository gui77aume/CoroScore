import matplotlib.pyplot as plt
from sklearn import svm
from sklearn_genetic.space import Categorical, Continuous, Integer
import Optim
import donnees3PROD as d
import evalModelFonction
import numpy as np
import easygui
import UI

effectif_test = np.round(UI.nsplit * (np.shape(d.donnees)[0]-d.n_outliers))
if np.alltrue(UI.features_patient) :
    print("utilisation des hyperparametres determinés pour 13 variables ")
    if UI.contamination_outliers == 'auto': # parametres optimisés pour environ 90 outliers autodetectés
        model = svm.SVC(C=79, gamma=0.16, kernel='poly', degree=3, class_weight='balanced', probability=True)
    else:
        # modele optimisé avec retrait  2% à 10% d'outliers
        model = svm.SVC(C=0.044, tol=0.00995, kernel='linear', class_weight='balanced', probability=True)

        # 10% model = svm.SVC(C=0.1419,tol=0.000450, kernel='linear', class_weight='balanced', probability=True)

    nom = f"ROC SVC hyperparamétré pour 13 variables\n" \
          f"effectif de test : {effectif_test}"
    evalModelFonction.eval_modele(model, d.X_train, d.y_train, d.X_test, d.y_test, nomModele=nom)


else:
    if UI.features_patient == UI.SOUS_ENSEMBLE_OPT:
        print("Jeu de 5 variables réduit connu, SVC hyperparamétre determinés pour 5 variables")
        model = svm.SVC(C=37.3,  kernel='linear',tol=0.006, class_weight='balanced', probability=True)

        nom = f"ROC SVC hyperparamétré pour 5 variables\n" \
              f"effectif de test (courbe rouge) : {effectif_test}"

        evalModelFonction.eval_modele(model, d.X_train, d.y_train,
                                      d.X_test, d.y_test,
                                      nomModele=nom)

    else:
        print("Jeu de variables jamais vu, apprentissage des meilleurs hyparametres")
        param_grid_svc = {
            'C': Continuous(0.01, 100),
            'gamma': Continuous(0.01, 10),
            'degree': Integer(2,3),
            'kernel': Categorical(['linear', 'rbf', 'sigmoid', 'poly']),
        }
        opt = Optim.Optim()
        opt.set_modele(svm.SVC(class_weight='balanced', probability=True), param_grid_svc)
        opt_mod = opt.run(d.X_train, d.y_train)
        model = opt_mod.best_estimator_
        b = opt_mod.best_params_
        nom = f"SVC sur jeu de variable réduit hyperparamétré par algorithme génétique\n" \
              f"avec {b}"
        evalModelFonction.eval_modele(model, d.X_train, d.y_train,
                                      d.X_test, d.y_test,
                                      nomModele=nom)
        print(f"\nMeilleurs parametres SVC FS : {opt_mod.best_params_}\n\n")

message = ""
# detection novelty
detecteur_outlier = svm.OneClassSVM(kernel='rbf', gamma=0.1, nu=UI.nu_novelty)
detecteur_outlier.fit(d.X_complet)
if detecteur_outlier.predict(d.X_patient) == -1:
    message += f"\nAttention : les données du patient sont très différentes des données d'apprentissage : " \
               f"la fonction de décision renvoie {detecteur_outlier.decision_function(d.X_patient)[0]:.2f}\n\n"


# apprentissage sur toutes les données
model.fit(d.X_complet, d.y_complet)

proba_coro = model.predict_proba(d.X_patient)[0, 0]
print("Traitement des données du patient (modèle appris sur l'ensemble des données) :")
if proba_coro >= 0.5:
    message += "Le modèle prédit la présence d'une coronaropathie"
else:
    message += "Le modèle prédit une absence de coronaropathie"

message += f"\n\nProbabilité de coronaropathie : {proba_coro:.2f}\n"
message += f"\nCliquer sur OK pour afficher les performances du modèle"
easygui.msgbox(message, title="Résultat")
plt.show()
