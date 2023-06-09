from bs4 import BeautifulSoup
import requests
import pandas as pd 
import os

dirname = os.path.dirname(__file__)

def webscraper(input_csv):
    
    df = pd.read_csv(dirname + '\\' + input_csv) 
    for ind in df.index:
        try:
            url = df['url'][ind]
            #print(url)
            HEADERS = {'User-Agent': 'Mozilla/5.0 (iPad; CPU OS 12_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148'} #to bypass webscraping detection
            source = requests.get(url, headers=HEADERS)
            soup = BeautifulSoup(source.text,'html.parser')
            #print(soup)
            content = soup.find('script', type = "application/ld+json")
            content = str(content)
            #ind_des = content.index('"description":') + len('"description":')
            #ind_des_next = content.index(',"review":') 
            #description_mov = content[ind_des+1:ind_des_next-1] #clamped string by 1 to remove quotations
            #print(description_mov)
            #df.loc[ind, 'description'] = description_mov

            ind_img = content.index('"image":') + len('"image":')
            ind_img_next = content.index(',"description":') 
            img_link = content[ind_img+1:ind_img_next-1]
            #print(img_link)
            df.loc[ind, 'image'] = img_link
            #print(df)
        except Exception as e:
            print(e)
    df.to_csv(dirname + '\\'+'best_bad_movies_with_descriptions_images.csv',)
webscraper('best_bad_movies_with_descriptions.csv')
