from bs4 import BeautifulSoup
import requests
import pandas as pd 
import os

dirname = os.path.dirname(__file__)
HEADERS = {'User-Agent': 'Mozilla/5.0 (iPad; CPU OS 12_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148'}
NUM_REVIEWS = 4

def scrape_reviews(input_csv):
    df = pd.read_csv(dirname + '/' + input_csv)

    id_lst = []
    content_lst = []
    link_lst = []

    for ind in df.index:
        try:
            url = df['url'][ind] + "reviews?ref_=tt_urv"
            #print(url)
            source = requests.get(url, headers=HEADERS)
            soup = BeautifulSoup(source.text, 'html.parser')
            reviews = soup.find_all("div", {"class":"review-container"})
            #print(reviews[0])
            #break
            for rev in range(NUM_REVIEWS):
                link = reviews[rev].find("a", {"class":"title"})["href"]
                print(link)
                content = reviews[rev].find("div", {"class":"text"}).contents[0]
                id_lst.append(df['const'][ind])
                link_lst.append(link)
                content_lst.append(content)
                #rev_lst.append({"movie_id":df["const"][ind], "review_link": link, "review_desc":content})
            #print(content)                
        except:
            continue
    
    df_dict = {"movie_id":id_lst, "link":link_lst, "review":content_lst}
    new_df = pd.DataFrame(df_dict)
    new_df.to_csv(dirname + "/" + "bad_movie_reviews.csv")


scrape_reviews('best_bad_movies_with_descriptions.csv')