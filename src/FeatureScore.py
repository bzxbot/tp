import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

class FeatureScore:
    def evaluate_features(self, X, y):
        bestfeatures = SelectKBest(score_func=f_classif, k=10)
        fit = bestfeatures.fit(X,y)
        dfscores = pd.DataFrame(fit.scores_)
        dfcolumns = pd.DataFrame(X.columns)
        #concat two dataframes for better visualization 
        featureScores = pd.concat([dfcolumns,dfscores],axis=1)
        featureScores.columns = ['Specs','Score']  #naming the dataframe columns
        print(featureScores.nlargest(10,'Score'))  #print 10 best features