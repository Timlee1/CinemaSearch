import json
import os
import re
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# These are the DB credentials for your OWN MySQL
# Don't worry about the deployment credentials, those are fixed
# You can use a different DB name if you want to

# MYSQL_USER_PASSWORD = "MayankRao16Cornell.edu"
MYSQL_USER = "root"
MYSQL_PORT = 3306
MYSQL_DATABASE = "badmoviesdb"
# Change to your password. Default: CinemaSearch
MYSQL_USER_PASSWORD = "CinemaSearch"

mysql_engine = MySQLDatabaseHandler(MYSQL_USER,MYSQL_USER_PASSWORD,MYSQL_PORT,MYSQL_DATABASE)

# Path to init.sql file. This file can be replaced with your own file for testing on localhost, but do NOT move the init.sql file
#mysql_engine.load_file_into_db()

app = Flask(__name__)
CORS(app)



# Sample search, the LIKE operator in this case is hard-coded, 
# but if you decide to use SQLAlchemy ORM framework, 
# there's a much better and cleaner way to do this
#
# terms is a tokenized list of strings representing key terms
# genres is a tokenized list of selected genres
# bounds is a double pair representing the bounds of the IMDB score range
def sql_search(input, genres, bounds):
    #print(input)
    genres_listing = ["horror","action","mystery","romance","sci-fi","western","drama","sci-fi","comedy","fantasy","crime","thriller","adventure","sport","biography","documentary"]
    tokens = tokenize(input)
    genres_lst = list()
    key_terms = list()

    for token in tokens:
        if token in genres_listing:
            genres_lst.append(token)
        else:
            key_terms.append(token)

    # query_sql = f"""SELECT imdb_rating,title,description,directors FROM movies WHERE LOWER( title ) LIKE '%%{movie.lower()}%%' limit 10"""
    query_sql = f"""SELECT imdb_rating,title,description,directors,genres FROM movies"""
    keys = ["imdb_rating","title","description","directors", "genres"]
    data = mysql_engine.query_selector(query_sql)
    dump = json.dumps([dict(zip(keys,i)) for i in data])

    # List of Dictionary of the movies retrieved. 
    movies = json.loads(dump)
    movies = tokenize_movies(movies)

    j_sim = jac_sim(genres_lst, movies)
    j_rankings = sorted(j_sim, key=lambda x: x[1], reverse=True)

    ed_dist = edit_dist(input,movies)    
    ed_dist_rankings = sorted(ed_dist, key=lambda x: x[1], reverse=False)

    #print([i[1] for i in rankings])
    return [i[0] for i in j_rankings]


# Tokenize some string
#
# @param text string to be tokenized
#
# @returns list of strings
def tokenize(text):
     return re.findall(r'[a-z\-]+', text.lower())


# Tokenize genres and descriptions of movies
#
# @param list of dictornaries of movies returned
#
# @returns list of movie 

def tokenize_movies(movies):
    out = list()
    for movie in movies:
        movie["terms"] = tokenize(movie["description"])
        movie["genre_tokens"] = tokenize(movie["genres"])
        out.append(movie)
    
    return out


# TODO Implement Jaccard Similarity
# Performs Jaccard simularity for genres
# 
# @param input: list of strings representing the genres
# @param movies: list of dictionaries for the the movies 
#
# @returns a list of pairs with each movie paired with its jaccard sim with query
def jac_sim(input,movies):
    A = set(input)
    out = list()
    for movie in movies:
        B = set(movie["genre_tokens"])
        out.append((movie, len(A.intersection(B)) / (1 + len(A.union(B)))))
    return out    

# TODO Implement Jaccard Similarity
# Performs cosine simularity for terms
# 
# @param input: list of strings representing the key terms
# @param movies: list of dictionaries for the the movies 
#
# @returns a list of pairs with each movie paired with its cosine sim with query
def cos_sim(input,movies):
   raise NotImplementedError

# TODO Implement Edit Distance
# Performs edit distance from input to movie name
# @param input: query that the user types into the search bar 
# @param movies: list of dictionaries for the the movies 
#
# @returns a list of pairs with each movie paired with its edit distance with query
def edit_dist(input,movies):
    del_cost = 1
    ins_cost = 1
    sub_cost = 2
    res = []
    input = input.lower()
    for movie in movies:
        title = movie['title'].lower()
        #print(title, input)
        dist = edit_distance(input,title, del_cost, ins_cost, sub_cost)
        #print(dist)
        res.append((movie,dist))
    #print(res)
    return res

def edit_distance(input, title, del_cost, ins_cost, sub_cost):
    m = len(input) + 1
    n = len(title) + 1

    chart = {(0, 0): 0}
    for i in range(1, m): 
        chart[i,0] = chart[i-1, 0] + del_cost 
    for j in range(1, n): 
        chart[0,j] = chart[0, j-1] + ins_cost
    for i in range(1, m):
        for j in range(1, n):
            if input[i-1] == title[j-1]:
                sub_cost_val = 0
            else:
                sub_cost_val = sub_cost
            chart[i, j] = min(
                chart[i-1, j] + del_cost,
                chart[i, j-1] + ins_cost,
                chart[i-1, j-1] + sub_cost_val
            )
    return chart[(len(input),len(title))]


@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/episodes")
def episodes_search():
    text = request.args.get("title")
    return sql_search(text, "", "")

# app.run(debug=True)
#print(tokenize_movies(sql_search("help")))
