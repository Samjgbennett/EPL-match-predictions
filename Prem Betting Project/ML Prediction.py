import pandas as pd
from Scraping import match_df

matches = match_df

#matches = pd.read_csv("matches.csv", index_col=0) # index is already in the data

#just having a look at the data
#matches.head()

#matches.shape

#seeing why there is less games than teams
#matches["team"].values_counts()

#investigating why liverpools is wrong, missing one season for liverpool
#matches[matches["team"] == "Liverpool"].sort_values("date")

#matches["round"].value_counts()



#Cleaning data



#machine learning can only work with float64 or int64, does not work with objects
#matches.dtypes

del matches["comp"]
del matches["notes"]

#overriding existing col with datetime data
matches["date"] = pd.to_datetime(matches["date"])
matches["target"] = (matches["result"] == "W").astype("int")
#print(matches)



#Creating predictors

#away or home game, converting to a number so can be used in algo, .cat.codes converts to integer

matches["venue_code"] = matches["venue"].astype("category").cat.codes

#Unique code for each opponent
matches["opp_code"] = matches["opponent"].astype("category").cat.codes

#we want to just have the hour rather than hour:min, so is an int, need a number input
matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")

#changing each day of the week to a number
matches["day_code"] = matches["date"].dt.dayofweek
#matches

#We want the target to be to predict this, change this to a number as well, 0 if lost or drew and 1 otherwise
matches["target"] = (matches["result"] == "W").astype("int")



#create inital ML model


#Good as non-linear model, linear would get confused with codes of teams etc
from sklearn.ensemble import RandomForestClassifier

#min_samples_split is the number of samples in a leaf of decision tree before splitting the mode, over and underfitting
#random_state is the seed
rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)

#has to be past predicting the future
train = matches[matches["date"] < '2022-01-01']

test = matches[matches["date"] > '2022-01-01']

predictors = ["venue_code", "opp_code", "hour", "day_code"]

#trains the rf using these predictors for the target
rf.fit(train[predictors], train["target"])

RandomForestClassifier(min_samples_split=10, n_estimators=50, random_state=1)

#make a prediction
preds = rf.predict(test[predictors])

#decide how accurate
from sklearn.metrics import accuracy_score

#create an accuracy score
error = accuracy_score(test["target"], preds)


#See what happens
combined = pd.DataFrame(dict(actual=test["target"], predicted=preds))
#two way table that shows when we predicted 0 or 1 what actually happened
pd.crosstab(index=combined["actual"], columns=combined["predicted"])


#different accuracy metric, 
from sklearn.metrics import precision_score

#when predicting a win what percentage were actual a win
precision_score(test["target"], preds)

#Improving Precision with rolling averages

#create a df for every squad
grouped_matches = matches.groupby("team")

#gives us a single group for Manchester City
group = grouped_matches.get_group("Manchester City").sort_values("date")


#takes a group and cols we want to compute rolling averages for and then takes in a set of new cols that we assign rolling averges to
def rolling_averages(group, cols, new_cols):
    group = group.sort_values("date") #sort by date
    rolling_stats = group[cols].rolling(3, closed='left').mean() #takes cols and compute rolling averages for those columns, take current week out
    group[new_cols] = rolling_stats 
    group = group.dropna(subset=new_cols) #drop any missing values, if no data available
    return group


#goals for, goals against, shots,shots on target, distance, freekicks, penalty goals, penalty kick attempts
cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
new_cols = [f"{c}_rolling" for c in cols] #add rolling on at end of each thing


#calling it for man city, as set prev
#rolling_averages(group, cols, new_cols)

#apply to all of our matches, .apply applies the rolling averages to all of them, creates a df for each team
matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x, cols, new_cols))

#dont need extra index level, we can now call each row
matches_rolling = matches_rolling.droplevel('team')

#we want a unique index, assign new indices, better for calling them
matches_rolling.index = range(matches_rolling.shape[0])



#Retraining our machine learning model



def make_predictions(data, predictors):
    train = data[data["date"] < '2022-01-01']
    test = data[data["date"] > '2022-01-01']
    rf.fit(train[predictors], train["target"])
    preds = rf.predict(test[predictors])
    combined = pd.DataFrame(dict(actual=test["target"], predicted=preds), index=test.index)
    error = precision_score(test["target"], preds)
    return combined, error
#call this fn
combined, error = make_predictions(matches_rolling, predictors + new_cols)

#print(precision)
#we have improved the precision

#adding columns so we can understand data a bit more, adding date, team, opponent and result
combined = combined.merge(matches_rolling[["date", "team", "opponent", "result"]], left_index=True, right_index=True)

#having a look at the data
combined.head(10)



#Combining Home and Away predictions



#normalising data, the map method will ignore missing values
class MissingDict(dict):
    __missing__ = lambda self, key: key

#Changing mapping values Brighton and Hove Albion -> Brighton etc
map_values = {"Brighton and Hove Albion": "Brighton", "Manchester United": "Manchester Utd", "Newcastle United": "Newcastle Utd", "Tottenham Hotspur": "Tottenham", "West Ham United": "West Ham", "Wolverhampton Wanderers": "Wolves"} 

#creates an instance of missing dicts
mapping = MissingDict(**map_values)

#mapping["West Ham United"] will return west ham

#adding new_team col with nicely done names
combined["new_team"] = combined["team"].map(mapping)

#now merge data frame with itself now all names line up, Arsenal vs Burnley = Burnley vs Arsenal etc
merged = combined.merge(combined, left_on=["date", "new_team"], right_on=["date", "opponent"])

#Look at things where a team was prediced to win and the other was predicted to lose vs the actual
#Accuracy at 67.5%

merged[(merged["predicted_x"] == 1) & (merged["predicted_y"] ==0)]["actual_x"].value_counts()

#looking at all the data points available
#print(matches.columns)