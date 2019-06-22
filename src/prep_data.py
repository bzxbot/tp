import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

data = pd.read_csv("data\caso-dengue2018.csv", sep=';') 

data.dropna(subset=["tp_classificacao_final"], inplace=True)

print(data["tp_classificacao_final"])

print(data.info())

X = data.drop(columns=["tp_classificacao_final"]).select_dtypes(exclude=['object']).fillna(0)  #independent columns
y = data["tp_classificacao_final"]    #target column i.e price range

print(X.info())

#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(10,'Score'))  #print 10 best features