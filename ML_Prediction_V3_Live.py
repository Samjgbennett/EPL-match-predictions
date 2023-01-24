import pandas as pd
import os
import numpy as np
#from Scraping import match_df


matches = pd.read_csv("All_Leagues2.csv") # index is already in the data
new_games=pd.read_csv("new_games.csv")

new_matches=new_games

matches=matches.replace({'FTR': {'A' : 0.0 ,'D' :1.0, 'H': 2.0}})
matches=matches.replace({'HTR': {'A' : 0.0 ,'D' :1.0, 'H': 2.0}})


new_matches=new_matches.replace({'FTR': {'A' : 0.0 ,'D' :1.0, 'H': 2.0}})


#print(matches)


#Creating predictors

#away or home game, converting to a number so can be used in algo, .cat.codes converts to integer



#Unique code for each opponent
matches["HomeTeam"] = matches["HomeTeam"].astype("category").cat.codes
matches["AwayTeam"] = matches["AwayTeam"].astype("category").cat.codes

new_matches["HomeTeam"] = new_matches["HomeTeam"].astype("category").cat.codes
new_matches["AwayTeam"] = new_matches["AwayTeam"].astype("category").cat.codes



#we want to just have the hour rather than hour:min, so is an int, need a number input



#We want the target to be to predict this, change this to a number as well, 0 if lost, 1 draw and 2 win



#data = df
data = matches

#print(data)
#print("dtypes")
#print(data.dtypes)
#print(data)

#all column names
cols = ['Date','HomeTeam','AwayTeam','FTHG','FTAG','FTR','HTHG','HTAG','HTR','HS','AS','HST','AST','HF','AF','HC',
'AC','HY','AY','HR','AR','B365H','B365D','B365A','BWH','BWD','BWA','IWH','IWD','IWA','PSH','PSD','PSA','WHH','WHD',
'WHA','VCH','VCD','VCA','MaxH','MaxD','MaxA','AvgH','AvgD','AvgA','B365>2.5','P<2.5','Max>2.5','Max<2.5','Avg>2.5',
'Avg<2.5','AHh','B365AHH','B365AHA','PAHH','PAHA','MaxAHH','MaxAHA','AvgAHH','B365CD','B365CA','BWCH','BWCD','BWCA',
'IWCH','IWCD','IWCA','PSCH','PSCD','PSCA','WHCH','WHCD','WHCA','VCCH','VCCD','VCCA','MaxCH','MaxCD','MaxCA','AvgCH',
'AvgCD','AvgCA','B365C>2.5','B365C<2.5','PC>2.5','PC<2.5','MaxC>2.5','MaxC<2.5','AvgC>2.5','AvgC<2.5','AHCh','B365CAHH',
'B365CAHA','PCAHH','PCAHA','MaxCAHH','MaxCAHA','AvgCAHH','AvgCAHA']

cols3 = ['Date','HomeTeam','AwayTeam','FTHG','FTAG','FTR','HTHG','HTAG','HTR','HS','AS','HST','AST','HF','AF','HC',
'AC','HY','AY','HR','AR','B365H','B365D','B365A','BWH','BWD','BWA','IWH','IWD','IWA','WHH','WHD',
'WHA','VCH','VCD','VCA','AvgH','AvgD','AvgA','B365>2.5','P<2.5','Max>2.5','Max<2.5','Avg>2.5',
'Avg<2.5','AHh','B365AHH','B365AHA','PAHH','PAHA','MaxAHH','MaxAHA','AvgAHH','B365CD','B365CA','BWCH','BWCD','BWCA',
'IWCH','IWCD','IWCA','PSCH','PSCD','PSCA','WHCH','WHCD','WHCA','VCCH','VCCD','VCCA','MaxCH','MaxCD','MaxCA','AvgCH',
'AvgCD','AvgCA','B365C>2.5','B365C<2.5','PC>2.5','PC<2.5','MaxC>2.5','MaxC<2.5','AvgC>2.5','AvgC<2.5','AHCh','B365CAHH',
'B365CAHA','PCAHH','PCAHA','MaxCAHH','MaxCAHA','AvgCAHH','AvgCAHA']




cols2 = ['Date','HomeTeam','AwayTeam','FTHG','FTAG','FTR','HTHG','HTAG','HTR','HS','AS','HST','AST','HF','AF','HC',
'AC','HY','AY','HR','AR','B365H','B365D','B365A']



cols4 = ['Date','HomeTeam','AwayTeam','FTR','B365H','B365D','B365A']

cols5 = ['FTR','B365H','B365D','B365A']

og_max = data.max(axis=0)

og_min = data.min(axis=0)



print(new_matches)
data = data[cols5].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

#data = data[cols5]
new_matches = new_matches[cols5].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

#new_matches = new_matches[cols5]

data = data.dropna(axis=0)

#Good as non-linear model, linear would get confused with codes of teams etc

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
import xgboost as xgb


#min_samples_split is the number of samples in a leaf of decision tree before splitting the mode, over and underfitting
#random_state is the seed
#rf = GradientBoostingRegressor(n_estimators=50, min_samples_split=10, random_state=1)
#rf = HistGradientBoostingRegressor(loss='absolute_error',learning_rate=0.08, random_state=1)
rf = HistGradientBoostingRegressor(learning_rate=0.08, random_state=1)


rf2 = xgb.XGBClassifier(objective="multi:softprob", random_state=30)

rf3 = HistGradientBoostingRegressor(loss='absolute_error',learning_rate=0.08, random_state=1)

matches = data
#has to be past predicting the future
#train = matches[matches["date"] < '2022-01-01']
#train = matches[matches["season"] != 1.0]


x_train = matches.drop("FTR",axis=1)
y_train = matches["FTR"]
x_test = new_matches.drop("FTR",axis=1)
y_test = new_matches["FTR"]


print(x_train)
print(x_test)
print(y_train)
print(y_test)


y_train2 = y_train*2
y_test2 = y_test*2

"""
y_train2 = y_train
y_test2 = y_test
"""

predictors = cols5

#trains the rf using these predictors for the target
rf = rf.fit(x_train, y_train)
rf2 = rf2.fit(x_train, y_train2)
rf3 = rf3.fit(x_train, y_train)


true_test_result = y_test*2 #use for non_xgb
#true_test_result = y_test
true_test_result2 = y_test2

#GradientBoostingRegressor(min_samples_split=10, n_estimators=50, random_state=11)

#make a prediction
y_preds = rf.predict(x_test)
y_preds2 = rf2.predict(x_test)
y_preds3 = rf3.predict(x_test)
#new_preds = rf.predict(new_test[predictors])

true_preds = np.round(y_preds*2) #use for non-xgb
true_preds2 = np.round(y_preds2)
true_preds3 = np.round(y_preds3*2) 


"""
true_preds = np.round(y_preds)
true_preds2 = np.round(y_preds2)
true_preds3 = np.round(y_preds3)
#true_new_preds = np.round(new_preds*2)
"""

#replace the messy bits


#preds = rf.predict(test["result"])
#print("preds", preds)



#decide how accurate



#See which situations affected acccuracy
#combined = pd.DataFrame(dict(actual=true_test_result, predicted=true_preds))
combined = pd.DataFrame(dict(actual=true_test_result, histgboost=true_preds))
combined2 = pd.DataFrame(dict(actual=true_test_result2, xgboost=true_preds2))
combined3 = pd.DataFrame(dict(actual=true_test_result, histgboost_ae=true_preds3))

#matches = pd.DataFrame(matches, columns=cols_old)




#two way table that shows when we predicted 0 or 1 what actually happened
print(pd.crosstab(index=combined["actual"], columns=combined["histgboost"]))
print(pd.crosstab(index=combined2["actual"], columns=combined2["xgboost"]))
print(pd.crosstab(index=combined["actual"], columns=combined3["histgboost_ae"]))

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size


#combined.to_csv("GB_GW522_23.csv")

merged = pd.concat([combined, new_games], axis=1)
merged = pd.concat([merged, combined2], axis=1)
merged = pd.concat([merged, combined3], axis=1)

merged = merged.drop('actual', axis=1)



print(merged)


merged.to_csv("Ful_Spurs.csv")



"""



Notes on how to read result:

if histgboost predicts 0 then precision = 0.7, if histboost predicts 2 then precision 0.77 - however has bad recall

xgboost 1 has precision 0.37 (best), however will rarely call this

if the hgboosts has highest f-score for 1's, although has good recall (0.87) bad precision (0.29) (overcalls draws)

if the hgboosts_ae has slightly lower f-score for 1's, although has good recall (0.61) better precision (0.3) (overcalls draws)

Histgboost and Histgboost_ae will predict similar things


If histgboost predicts 0 or 2 trust it, will do this too few times,
Always choose a histgboost_ae 2 or 0 over a 1 for histgboost
If xgboost predicts a 1 best guess but also keep others in reason
If histgboost predicts a draw and the others dont technically math says its better to trust others due to higher f-score and accuracy
However use personal ideas for draws, inclusing attacking ability and relative league positions


Histgboost: 38%
0's - 70%, f = 0.2
1's - 29%, f = 0.43
2's - 77%, f = 0.41
xgboost: 52%
0's - 49%, f = 0.48
1's - 0.37%, f = 0.04
2's - 53%, f = 0.65
Histgboost_ae: 46%
0's - 60%, f = 0.38
1's - 30%, f = 0.4
2's - 66%, f = 0.57


If everytime we get a 1 we change to a 2,
accuracy increases to :

Histgboost: 49.5%
xgboost: 53%
Histgboost_ae: 51.7%

means: 2: 2.62
       1: 3.77
       0: 4.37


medians: 2: 2.2
        1: 3.4
        2: 3.4

Thus worst case scenario i need to win 45% of matches, I can guarantee over this

45% of 2's
29.4% of 1's
29.4% of 0's
"""