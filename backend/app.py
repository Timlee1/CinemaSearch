import json
import os
import re
import math
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
#from nltk.stem import PorterStemmer


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

#stemmers
#ps = PorterStemmer()


# Sample search, the LIKE operator in this case is hard-coded, 
# but if you decide to use SQLAlchemy ORM framework, 
# there's a much better and cleaner way to do this
#
# terms is a tokenized list of strings representing key terms
# genres is a tokenized list of selected genres
# bounds is a double pair representing the bounds of the IMDB score range
# filter is a string representing extra SQL queries
def sql_search(input, genres, bounds, filter):
    #print(input)
    genres_listing = ["horror","action","mystery","romance","sci-fi","western","drama","sci-fi","comedy","fantasy","crime","thriller","adventure","sport","biography","documentary"]
    tokens = tokenize(input)
    genres_lst = list()
    key_terms = list() # dictonary to track term freqs

    for token in tokens:
        if token in genres_listing:
            genres_lst.append(token)
        else:
            key_terms.append(token)
    
    key_terms = set(key_terms) #only count each term once in query

    # query_sql = f"""SELECT imdb_rating,title,description,directors FROM movies WHERE LOWER( title ) LIKE '%%{movie.lower()}%%' limit 10"""
    query_sql = f"""SELECT id,imdb_rating,title,description,directors,genres FROM movies {filter.lower()}"""
    keys = ["id","imdb_rating","title","description","directors", "genres"]
    data = mysql_engine.query_selector(query_sql)
    dump = json.dumps([dict(zip(keys,i)) for i in data])

    # List of Dictionary of the movies retrieved. 
    movies = json.loads(dump)
    movies = tokenize_movies(movies)

    j_sim = jac_sim(genres_lst, movies)
    c_sim = cos_sim(key_terms, movies)
    rankings = list()
    for pair in j_sim:
        id = pair[0]["id"]
        #infinite weight if name matches perfectly
        if input == pair[0]["title"].lower():
            rankings.append((pair[0], 100))
        # look up cosine simularity for same movie and combine simualrity measures, equal weight
        #don't include zero simularity
        else:
            if c_sim[id] + pair[1] > 0:
                rankings.append((pair[0], c_sim[id] + pair[1]))
    

    rankings = sorted(rankings, key=lambda x: x[1], reverse=True)
    #print([i[1] for i in rankings][:10])
    
    rankings = [i[0] for i in rankings][:10]

    return rankings
    

    #j_rankings = sorted(j_sim, key=lambda x: x[1], reverse=True)

    #ed_dist = edit_dist(input,movies)    
    #ed_dist_rankings = sorted(ed_dist, key=lambda x: x[1], reverse=False)

    #print([i[1] for i in rankings])
    #return [i[0] for i in j_rankings]



# Tokenize some string
#
# @param text string to be tokenized
#
# @returns list of strings
def tokenize(text):
    #out = list()
    tokens = re.findall(r'[a-z\-]+', text.lower())
    #for token in tokens:
     #   out.append(ps.stem(token)) #stem tokens
    #return out
    return tokens
         


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

#create and save term postings for movie descriptions, assume
#descriptions have been tokenized and have a "terms" section
# track movie id and term frequencies
# 
# Also builds document norms
# ONLY RUN WHEN NEW DATA IS ADDED
def term_postings(movies):
    postings = dict()
    norms = dict()
    for movie in movies:
        #build norms for each movie description
        norm = 0
        #first get term frequenices in description
        tf = dict()
        for term in movie["terms"]:
            #term appeared before in movie desc
            if term in tf:
                tf[term]+=1
            else:
                tf[term] = 1

        # now iterate through terms and add to postings and norm
        for term in tf:
            if term not in postings: #first time term, add to postings
                postings[term] = dict() #dict for quick id look up
            postings[term][movie["id"]] = tf[term]
            norm += tf[term]**2
        
        norm = math.sqrt(norm)
        norms[movie["id"]] = norm
    #dump into json for later reference
    json.dump(postings, open("term_postings.json", 'w'))
    json.dump(norms, open("doc_norms.json", 'w'))




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
   #load stored data
   term_postings = json.load(open("term_postings.json", 'r'))
   norms = json.load(open("doc_norms.json", 'r'))
   #build doc scores for each movie
   doc_scores = dict()
   for term in input:
       if term in term_postings:
           p = term_postings[term]
           # inverse doc frequency, add 1 incase length is 0
           idf = 1/ (len(p)+1)
           #build doc scores for term
           for movie_id in p:
               if movie_id in doc_scores:
                   #add term freq times idf to doc score, assume query weight for term is 1
                   doc_scores[movie_id] += (p[movie_id] * idf)
               else:
                   doc_scores[movie_id] = ((p[movie_id] * idf))
   
   #return dict with id's so we can quickly compare with jaccard
   out = dict()
   for movie in movies:
        #need to make string id temporailly due to json loading it as an int
        # store normally
        id = str(movie["id"])
        if  id in doc_scores:
            simularity = doc_scores[id] / (norms[id])
            out[movie["id"]] =  simularity
        else:
            out[movie["id"]] =  0 # 0 simularity if doc-score is non-existant
   return out
    
   
           
       


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
    return sql_search(text, "", "", "")
@app.route("/episodes/sort")
def episodes_sort():
    text = request.args.get("title")
    sort_type = request.args.get("sort")
    data = sql_search(text, "", "", "ORDER BY imdb_rating DESC")
    if(sort_type ==f 'asc'):
        data = sorted(data, key=lambda x: x['imdb_rating'], reverse=True)
    else:
        data = sorted(data, key=lambda x: x['imdb_rating'], reverse=False)
    return data

# app.run(debug=True)
#print(tokenize_movies(sql_search("help")))
