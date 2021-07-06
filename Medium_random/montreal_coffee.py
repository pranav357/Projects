import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import html5lib
import pgeocode
import seaborn as sns
import matplotlib.pyplot as plt
import folium

#Using WebScraping to extract data
url = 'https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_H'

# Use get to download the content of the webpage
html_data = requests.get(url).text

#Parse the html data using beautiful_soup.
soup = BeautifulSoup(html_data,"html5lib")

# Create a list
table_contents = []

table = soup.find('table')

#initialize an empty dictionary to save data in
postal_codes_dict = {}
for table_cell in soup.find_all('td'):
    try:
        postal_code = table_cell.p.b.text # get the postal code
        postal_code_investigate = table_cell.span.text
        neighborhoods_data = table_cell.span.text # get the rest of the data in the cell
        borough = neighborhoods_data.split('(')[0] # get the borough in the cell
        
        # if the cell is not assigned then ignore it
        if neighborhoods_data == 'Not assigned':
            neighborhoods = []
        # else process the data and add it to the dictionary
        else:
            postal_codes_dict[postal_code] = {}
            
            try:
                neighborhoods = neighborhoods_data.split('(')[1]
            
                # remove parantheses from neighborhoods string
                neighborhoods = neighborhoods.replace('(', ' ')
                neighborhoods = neighborhoods.replace(')', ' ')

                neighborhoods_names = neighborhoods.split('/')
                neighborhoods_clean = ', '.join([name.strip() for name in neighborhoods_names])
            except:
                borough = borough.strip('\n')
                neighborhoods_clean = borough
 
            # add only neighborhood to dictionary
            #postal_codes_dict[postal_code]['borough'] = borough
            postal_codes_dict[postal_code]['neighborhoods'] = neighborhoods_clean
    except:
        pass

# create an empty dataframe
columns = ['PostalCode', 'Neighborhood']
montreal_data = pd.DataFrame(columns=columns)
montreal_data

# populate dataframe with data from dictionary
for ind, postal_code in enumerate(postal_codes_dict):
    #borough = postal_codes_dict[postal_code]['borough']
    neighborhood = postal_codes_dict[postal_code]['neighborhoods']
    montreal_data = montreal_data.append({"PostalCode": postal_code, 
                                        "Neighborhood": neighborhood},
                                        ignore_index=True)

# print number of rows of dataframe
montreal_data.shape[0]

montreal_data.info()

#Get geographical coordinates
nomi = pgeocode.Nominatim('ca')
nomi.query_postal_code("H1A")

#Creating a function to use pgeocode

def get_latitude(postal):# get the latitude

  nomi = pgeocode.Nominatim('ca')
  response = nomi.query_postal_code(postal)
  return nomi.query_postal_code(postal)[9]


def get_longitude(postal):# get the longitude

  nomi = pgeocode.Nominatim('ca')
  response = nomi.query_postal_code(postal)
  return nomi.query_postal_code(postal)[10]

montreal_data['Latitude'] = montreal_data.loc[:,'PostalCode'].apply(lambda x: get_latitude(x))
montreal_data['Longitude'] = montreal_data.loc[:,'PostalCode'].apply(lambda x: get_longitude(x))

montreal_data.head()  

#See nulls
df1 = montreal_data[montreal_data.isna().any(axis=1)]
df1

#Remove the row
montreal_data = montreal_data.drop(68, axis=0)

sns.boxplot( x=montreal_data['Latitude'])
plt.show()

sns.boxplot( x=montreal_data['Longitude'])
plt.show()

#Remove outliers based on plots
montreal_data = montreal_data[(montreal_data['Latitude'] <= 80)] 
montreal_data = montreal_data[(montreal_data['Longitude'] <= -10)]

# Save the data to a csv file
montreal_data.to_csv('montreal_data.csv', index=False)

#Find areas of montreal with low density coffee shops
montreal_data.loc[montreal_data['Neighborhood']== 'McGill University']

# create map of Montreal using latitude and longitude values
latitude = 45.504
longitude = -73.5747

map_montreal = folium.Map(location=[latitude, longitude], zoom_start=11)

# add markers to map
for lat, lng, label in zip(montreal_data['Latitude'], montreal_data['Longitude'], montreal_data['Neighborhood']):
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_montreal)  
    
map_montreal.save('montreal.html')

#Use FourSquareAPI to explore and segment neighbourhoods

