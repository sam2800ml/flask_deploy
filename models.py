import pandas as pd
import numpy as np

from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import  GridSearchCV

from utils import Utils

class Models:

    def __init__(self):
        self.reg = {
            'SVR':SVR(),
            'Gradiente': GradientBoostingRegressor()
        }

        self.params = {
            'SVR' :{
                'kernel': ['linear', 'poly', 'rbf'],
                'C': [1,5, 10],
                'epsilon': [0.1, 1],
                'gamma':['auto','scale']
            },
            'Gradiente':{
                "learning_rate": [0.01, 0.05, 0.1],
                "loss": ['squared_error','absolute_error']
            }
        }
    
    def grid_training(self,X,y):
        best_score = 999
        best_model = None

        for name, reg in self.reg.items():
            print(name)
            grid_reg = GridSearchCV(reg, self.params[name],cv=3).fit(X,y.values.ravel())
            score = np.abs(grid_reg.best_score_)

            if score < best_score:
                best_score = score
                best_model = grid_reg

        
        utils = Utils()
        utils.model_export(best_model, best_score)

