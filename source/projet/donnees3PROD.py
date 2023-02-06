from sklearn.preprocessing import MinMaxScaler, RobustScaler
import numpy as np
from numpy import newaxis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
import UI

X_brut_patient = UI.X_brut_patient

donnees = np.loadtxt('/home/guillaume/TPPyCharm/projet/heart.dat', delimiter=' ', usecols=range(14))

# y ##############################################################

# variable prédite y : 1 si coro nécessaire 0 sinon
y_complet = donnees[:, 13] - 1

# Quantitatives ###################################################

# age du patient
age = donnees[:, 0, newaxis]
age_p = X_brut_patient[0]

# cholesterolémie en mg/dl
cholesterolemie = donnees[:, 4, newaxis]
cholesterolemie_p = X_brut_patient[4]

# pression arterielle lors de l'admission
pression_arterielle = donnees[:, 3, newaxis]
pression_arterielle_p = X_brut_patient[3]

# pouls max mesuré durant l'exercice
pouls_max = donnees[:, 7, newaxis]
pouls_max_p = X_brut_patient[7]

# valeur de la dpression ST induite par l'effort, comparée au repos
depression_st_induite = donnees[:, 9, newaxis]
depression_st_induite_p = X_brut_patient[9]

# nombre de vaisseaux majeurs obstrués à plus de 50%  apparents à la scintigraphie
n_vaisseaux = donnees[:, 11, newaxis]
n_vaisseaux_p = X_brut_patient[11]

X_quantitatives = np.hstack((age, pression_arterielle,
                             cholesterolemie, pouls_max, depression_st_induite,
                             n_vaisseaux,
                             ))
X_quantitatives_p = np.array((age_p, pression_arterielle_p, cholesterolemie_p, pouls_max_p, depression_st_induite_p,
                              n_vaisseaux_p))

scaler = RobustScaler()
scaler.fit(X_quantitatives)
X_quantitatives = scaler.transform(X_quantitatives)
X_quantitatives_p = scaler.transform(X_quantitatives_p.reshape(1, -1))[0]


# Donnees qualitatives  ####################################

def to_one_hot(colonne):
    encodeur = OneHotEncoder()
    # colonne = colonne[:, newaxis]
    encodeur.fit(colonne)
    return encodeur.transform(colonne).toarray()


# le patient est-il de sexe masculin
sexe_masculin = donnees[:, 1, newaxis]
sexe_masculin_p = X_brut_patient[1]

# douleur tho angor typique/ angor atypique/ autre douleur / asymptomatique
# type_douleur_thoracique = to_one_hot(donnees[:, 2] - 1)
# on inverse l'ordre des valeurs pour avoir gravité apparente croissante
type_douleur_thoracique = np.abs(donnees[:, 2] - 4)
type_douleur_thoracique /= 3
type_douleur_thoracique = type_douleur_thoracique[:, newaxis]

type_douleur_thoracique_p = np.abs(X_brut_patient[2] - 4)
type_douleur_thoracique_p /= 3

# if use_one_hot:
#     type_douleur_thoracique = to_one_hot(type_douleur_thoracique)

# glycémie à jeun supérieure à 120 mg/dL
glycemie_sup_120 = donnees[:, 5, newaxis]
glycemie_sup_120_p = X_brut_patient[5]

# type ecg au repos :
# 0: normal
# 1: ST-T wave abnormality (T wave inversions and/or ST elevation or depression >0.05mV)
# 2: showing probable or definite left ventricular hypertrophy by Estes' criteria.
# type_ecg_repos = to_one_hot(donnees[:, 6])
# on laisse par ordre de gravité apparente
type_ecg_repos = donnees[:, 6]
type_ecg_repos /= 2
type_ecg_repos = type_ecg_repos[:, newaxis]

type_ecg_repos_p = X_brut_patient[6]
type_ecg_repos_p /= 2

# if use_one_hot:
#     type_ecg_repos = to_one_hot(type_ecg_repos)

# l'exercice induit un angor
angor_induit_exe = donnees[:, 8, newaxis]

angor_induit_exe_p = X_brut_patient[8]

# pente du segment ST induit par exercice
# 1: upsloping, 2: flat, 3: downsloping.
# pente_st_induite = to_one_hot(donnees[:, 10] - 1)
pente_st_induite = np.abs(donnees[:, 10] - 3)  # gravité apparente à verifier
pente_st_induite /= 2
pente_st_induite = pente_st_induite[:, newaxis]

pente_st_induite_p = np.abs(X_brut_patient[10] - 3)  # gravité apparente à verifier
pente_st_induite_p /= 2

# if use_one_hot:
#     pente_st_induite = to_one_hot(pente_st_induite)

# thalium stress test 3: normal blood flow, 6: fixed defect, 7: reversible defect.
thal = donnees[:, 12] - 3  # on supprime la classe 0 implicite
thal = np.where(thal == 4, 6, thal)
thal /= 6  # normalisation
thal = thal[:, newaxis]

thal_p = X_brut_patient[12] - 3
if thal_p == 4:
    thal_p = 6
thal_p /= 6

# if use_one_hot:
#     thal = to_one_hot(thal)


X_qualitatives = np.hstack((sexe_masculin,
                            type_douleur_thoracique,
                            glycemie_sup_120,
                            type_ecg_repos,
                            angor_induit_exe,
                            pente_st_induite,
                            thal))

X_qualitatives_p = np.array((sexe_masculin_p,
                             type_douleur_thoracique_p,
                             glycemie_sup_120_p,
                             type_ecg_repos_p,
                             angor_induit_exe_p,
                             pente_st_induite_p,
                             thal_p))

X_complet = np.hstack((X_quantitatives, X_qualitatives))
X_patient = np.hstack((X_quantitatives_p, X_qualitatives_p)).reshape(1, -1)

X_complet = X_complet[:, UI.features_patient]
X_patient = X_patient[:, UI.features_patient]

# Suppression outliers #############################################
n_outliers = 0
if UI.remove_outliers:
    detecteur_outlier = IsolationForest(contamination=UI.contamination_outliers)
    detecteur_outlier.fit(X_complet)
    y_outliers = detecteur_outlier.predict(X_complet)
    n_outliers = y_outliers[y_outliers == -1].size
    print(f"nombre d'outliers retirés du jeu de données : {n_outliers}")
    X_complet = X_complet[y_outliers == 1]
    y_complet = y_complet[y_outliers == 1]

# Répartition train-test #############################################
X_train, X_test, y_train, y_test = train_test_split(X_complet, y_complet, test_size=UI.nsplit,
                                                    shuffle=True, random_state=0)
