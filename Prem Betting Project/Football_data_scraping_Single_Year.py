import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
import numpy as np

#scrape multiple games shots

base_url ='https://understat.com/league/EPL/'

match = str(input('Please enter the Year: '))
url = base_url + match


res = requests.get(url)
soup = BeautifulSoup(res.content,'lxml')
scripts = soup.find_all('script') #finding all the script tags in html

#get only the datesData
strings = scripts[1].string

#strip uneccessary symbols so only have json data

ind_start = strings.index("('")+2
ind_end = strings.index("')")

json_data = strings[ind_start:ind_end]
json_data = json_data.encode('utf8').decode('unicode_escape')

#convert string to json format
data = json.loads(json_data)

#print(data)

#Creating subsection of data

xg = []
goals = []
home = []
away = []


for index in range(len(data)):
    for key in data[index]:
        if key =='xG' :
            xg.append(data[index][key])
        if key =='goals':
            goals.append(data[index][key])
        if key =='h':
            home.append(data[index][key])
        if key =='a':
            away.append(data[index][key])


        
col_names1 = ['Home','Away','Goals','xG']
df = pd.DataFrame([home,away,goals,xg],index=col_names1)
df = df.T

#print(df)


#import numpy as np
#np.savetxt('scores.csv', [p for p in zip(df)], delimiter=',', fmt='%s')


