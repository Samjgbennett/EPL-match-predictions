#Scraping our first page with requests

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

#list of years we want
years = list(range(2022, 2020, -1))
#will contain match log for one team in one season, will combine these to a big dataframe at the end
all_matches = []

standings_url = "https://fbref.com/en/comps/9/11160/2021-2022-Premier-League-Stats"


import time


for year in years:
    data = requests.get(standings_url)
    soup = BeautifulSoup(data.text,'lxml')
    standings_table = soup.select('table.stats_table')[0]

    links = [l.get("href") for l in standings_table.find_all('a')]
    links = [l for l in links if '/squads/' in l]
    team_urls = [f"https://fbref.com{l}" for l in links]
    
    #grabbing url for prev season
    previous_season = soup.select("a.prev")[0].get("href")
    standings_url = f"https://fbref.com{previous_season}"
    
    #loop through each team indvidual url and get match logs
    for team_url in team_urls:
        #getting rid of additonal stuff in the url, stuff before the slash and then getting rid of -stats and -
        team_name = team_url.split("/")[-1].replace("-Stats", "").replace("-", " ")
        
        data = requests.get(team_url)
        matches = pd.read_html(data.text, match="Scores & Fixtures")[0]
        
        soup = BeautifulSoup(data.text,'lxml')
        links = [l.get("href") for l in soup.find_all('a')]
        links = [l for l in links if l and 'shooting/' in l]
        time.sleep(3)
        data = requests.get(f"https://fbref.com{links[0]}")
        time.sleep(3)
        shooting = pd.read_html(data.text, match="Shooting")[0]
        shooting.columns = shooting.columns.droplevel()
        #merge shooting stats with match stats

        #sometimes shooting stats arent available, so sometimes get error, if error then ignore
        try:
            team_data = matches.merge(shooting[["Date", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]], on="Date")
        except ValueError:
            continue
        #only keep pl data
        team_data = team_data[team_data["Comp"] == "Premier League"]
        #add in the season and team cols
        team_data["Season"] = year
        team_data["Team"] = team_name
        #add team data to that list
        all_matches.append(team_data)
        #does nothing for 1 second, websites dont want you to scrape too quickly, so this avoids blocking
        time.sleep(2)

#len(all_matches)

#combine individual df to one df
match_df = pd.concat(all_matches)
#make all column name lower case so easier to grab columns
match_df.columns = [c.lower() for c in match_df.columns]



#Final match reults



#print(match_df)

#write data to csv file
#match_df.to_csv("matches.csv")
