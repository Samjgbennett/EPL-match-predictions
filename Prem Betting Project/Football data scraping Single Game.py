import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
import numpy as np

#scrape the single game shots

base_url ='https://understat.com/match/'

match = str(input('Please enter the match id: '))
url = base_url + match


res = requests.get(url)
soup = BeautifulSoup(res.content,'lxml')
scripts = soup.find_all('script') #finding all the script tags in html

#get only the shotsData
strings = scripts[1].string

#strip uneccessary symbols so only have json data

ind_start = strings.index("('")+2
ind_end = strings.index("')")

json_data = strings[ind_start:ind_end]
json_data = json_data.encode('utf8').decode('unicode_escape')

#convert string to json format
data = json.loads(json_data)


#create lists 
h_xg = []
a_xg = []

h_goals = []
a_goals = []

data_away = data['a'] #away data
data_home = data['h'] #home data

h_team = []
a_team = []

for index in range(1):
    for key in data_home[index]:
        if key =='h_team' :
            h_team.append(data_home[index][key])

for index in range(1):
    for key in data_away[index]:
        if key =='a_team' :
            a_team.append(data_away[index][key])

h_team = str(h_team)
a_team = str(a_team)

for index in range(len(data_home)):
    for key in data_home[index]:
        if key =='xG' :
            h_xg.append(data_home[index][key])
        if ((key =='h_goals') & (index != index-1)) :
            h_goals.append(data_home[index][key])
        
        
for index in range(len(data_away)):
    for key in data_away[index]:
        if key =='xG' :
            a_xg.append(data_away[index][key])
        if key =='a_goals' :
            a_goals.append(data_away[index][key])
        

#create the dataframe
col_names1 = ['h_xg','h_goals']
col_names2 = ['a_xg','a_goals']
h_df = pd.DataFrame([h_xg,h_goals],index=col_names1)
h_df = h_df.T
a_df = pd.DataFrame([a_xg,a_goals],index=col_names2)
a_df = a_df.T

#turn into float so can do maths
h_df = h_df.astype(float)
a_df = a_df.astype(float)

#find the sum of xg
sum_h_xg = h_df["h_xg"].sum()
sum_a_xg = a_df["a_xg"].sum()


#find number of goals in the match
match_h_goals = h_df["h_goals"].mean()
match_a_goals = a_df["a_goals"].mean()


#create final df
col_names_hf = ['Home team','Home Goals','Home xG']
col_names_af = ['Away team','Away Goals','Away xG']

hf_df = pd.DataFrame([h_team,match_h_goals,sum_h_xg],index=col_names_hf)
af_df = pd.DataFrame([a_team,match_a_goals,sum_a_xg],index=col_names_af)

#merge home and away tables
results = [hf_df,af_df]
f_df = pd.concat(results)
print(f_df)
