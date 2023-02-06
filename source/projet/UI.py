
import numpy as np

# Zone de saisie de valeurs par utilisateur, à remplacer par une UI graphique....
# Données du patient ###############################################
# Si valeur manquante, saisir 0 et mettre à jour le masque features_patient
# X_brut_patient = np.array([67.0, 0.0, 3.0, 115.0, 564.0, 0.0, 2.0, 160.0, 0.0, 1.6, 2.0, 0.0, 7.0]) # absence coro
# X_brut_patient = np.array([57.0, 1.0, 2.0, 124.0, 261.0, 0.0, 0.0, 141.0, 0.0, 0.3, 1.0, 0.0, 7.00]) # présence
X_brut_patient = np.array([20.0, 0.0, 2.0, 124.0, 261.0, 0.0, 0.0, 141.0, 0.0, 0.3, 1.0, 0.0, 7.00])  # outlier
# (femme de 20 ans avec constantes homme 57 ans)
X_brut_patient.reshape(1, -1)

# Selection de variables (voir liste plus bas) ###########################################
TOUTES_VARIABLES = [True, True, True, True, True, True, True, True, True, True, True, True, True]
# [False, False, False, True, False, True, True, True, False, False, True, True, True]
SOUS_ENSEMBLE_OPT = [False, False, False, False,  True,  True, False,  True, False,  True, False, False, True]
features_patient_autre = [True, True, True, True, True, True, True, True, True, True, True, True, False]
# masque utilisé dans le code, à paramétrer selon éventuelles valeurs manquantes ou feature selection
features_patient = TOUTES_VARIABLES

use_one_hot = False  # utiliser ou non l'encodage one hot pour les variables qualitatives Pour tests seulement
# TODO utiliser un encodeur one hot en une seule fois si one_hot retenu
nsplit = 0.3  # taux de donnees à conserver pour tester le modèle
remove_outliers = True  # detecter et supprimer automatiquement les observations les plus extremes
contamination_outliers = 0.02  # taux d'outliers à retirer: auto (env 90 patients !) ou 0.1 (27 patients)
nu_novelty = 0.02

# FIN DU PARAMETRAGE PAR UI

#       -- 1. age
#       -- 2. sex
#       -- 3. chest pain type  (4 values)
#       -- 4. resting blood pressure
#       -- 5. serum cholestoral in mg/dl
#       -- 6. fasting blood sugar > 120 mg/dl
#       -- 7. resting electrocardiographic results  (values 0,1,2)
#       -- 8. maximum heart rate achieved
#       -- 9. exercise induced angina
#       -- 10. oldpeak = ST depression induced by exercise relative to rest
#       -- 11. the slope of the peak exercise ST segment
#       -- 12. number of major vessels (0-3) colored by flourosopy
#       -- 13.  thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
