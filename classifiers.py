#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exemplo de clasificadores
@author: diegobertolini
"""
import sys
import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split

def main():

        # load data
        print ("Loading data...")
        dados = np.loadtxt('seed_obif_f1f2f3.txt') ;
        xc = dados[:,1:-1]
        yc = dados[:, -1]
        
        x_train, x_test, y_train, y_test = train_test_split(xc, yc, test_size=0.3)

        ss = StandardScaler()
        x_train = ss.fit_transform(x_train)
        x_test = ss.transform(x_test)
        
# k-NN classifier
        #from sklearn.metrics import classification_report
        #neigh = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
        #neigh.fit(x_train, y_train)
        #neigh.score(X_test, y_test)
        #print(classification_report(y_test, neigh.predict(x_test)))
        
#SVM com Grid search
        #https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
        C_range = 2. ** np.arange(-5,15,2)
        gamma_range = 2. ** np.arange(3,-15,-2)
        # instancia o classificador, gerando probabilidades
        srv = svm.SVC(probability=True, kernel='rbf') # kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
        ss = StandardScaler()
        pipeline = Pipeline([ ('scaler', ss), ('svm', srv) ])
        
        param_grid = {
            'svm__C' : C_range,
            'svm__gamma' : gamma_range
        }
        
        # faz a busca
        grid = GridSearchCV(pipeline, param_grid, n_jobs=-1, verbose=True)
        grid.fit(x_train, y_train)
        
        # recupera o melhor modelo
        model = grid.best_estimator_
        print(classification_report(y_test, model.predict(x_test)))
#        
### MLP 
#        scaler = StandardScaler()
#        X_train = scaler.fit_transform(X_train)
#        X_test = scaler.fit_transform(X_test)
#        
#        clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(100, 100, 100), random_state=1)
#        clf.fit(X_train, y_train)
#        print(clf.predict(X_test))
#        print(classification_report(y_test, clf.predict(X_test)))
        
if __name__ == "__main__":
        if len(sys.argv) != 1:
                sys.exit("")

        main()