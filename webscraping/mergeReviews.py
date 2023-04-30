import pandas as pd 
import os

dirname = os.path.dirname(__file__)

def mergeReviews():
    df = pd.read_csv(dirname + '\\'+'best_bad_movies_with_descriptions_images.csv')
    print(df)
    df_reviews = pd.read_csv(dirname + '\\'+'bad_movie_reviews.csv')
    for ind in df.index:
        movie_id = df['const'][ind]
        mov_reviews = df_reviews.loc[df_reviews['movie_id'] == movie_id]
        mov_merged_reviews = ""
        for ind_reviews in mov_reviews.index:
          rev = mov_reviews['review'][ind_reviews] + " "
          mov_merged_reviews += rev 
        df.loc[ind, 'reviews'] = mov_merged_reviews
    
    df.to_csv(dirname + '\\'+'best_bad_movies_with_descriptions_images_reviews.csv',)
        

mergeReviews()