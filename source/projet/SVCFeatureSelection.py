import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import accuracy_score
from sklearn_genetic import GAFeatureSelectionCV
from sklearn_genetic.plots import plot_fitness_evolution

import donnees3PROD as d

#  1 recherche de combinaison de varibles avec meilleur SVC 13 vars
model = svm.SVC(C=0.044, tol=0.00995, kernel='linear', class_weight='balanced', probability=True)
model.fit(d.X_train, d.y_train)

y_test_predit = model.predict(d.X_test)
print(accuracy_score(d.y_test, y_test_predit))

svc_disp = RocCurveDisplay.from_estimator(model, d.X_test, d.y_test)
plt.show()

evolved_estimator = GAFeatureSelectionCV(
    estimator=model,
    cv=5,
    scoring="accuracy",
    population_size=30,
    generations=30,
    n_jobs=-1,
    verbose=True,
    keep_top_k=2,
    elitism=True,
)

evolved_estimator.fit(d.X_train, d.y_train)
features = evolved_estimator.best_features_

y_predict_ga = evolved_estimator.predict(d.X_test[:, features])
accuracy = accuracy_score(d.y_test, y_predict_ga)

print(features)
print("accuracy : ", accuracy)


plot_fitness_evolution(evolved_estimator)
plt.show()

