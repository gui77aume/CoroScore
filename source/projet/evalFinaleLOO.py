from sklearn import svm
import donnees3PROD as d
import numpy as np
import UI
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut

if UI.features_patient == UI.TOUTES_VARIABLES:
    model = svm.SVC(C=0.044, tol=0.00995, kernel='linear', class_weight='balanced', probability=True)
else: #SOUS ENSEMBLE OPTIMAL
    model = svm.SVC(C=37.3, kernel='linear', tol=0.006, class_weight='balanced', probability=True)

cv = LeaveOneOut()
scores = cross_val_score(model, d.X_complet, d.y_complet, scoring='accuracy',
                         cv=cv, n_jobs=-1)
print(np.mean(scores))
