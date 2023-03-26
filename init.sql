SET GLOBAL local_infile=1;
CREATE DATABASE IF NOT EXISTS badmoviesdb;
USE badmoviesdb;
DROP TABLE IF EXISTS movies;
CREATE TABLE movies(
    id INT,
    const TEXT,
    created TEXT,
    modified TEXT,
    description TEXT,
    title varchar(64),
    url TEXT,
    title_type varchar(64),
    imdb_rating REAL,
    runtime INT,
    year INT,
    genres TEXT,
    num_votes INT,
    release_date TEXT,
    directors TEXT
);

LOAD DATA LOCAL INFILE 'best_bad_movies.csv' 
INTO TABLE movies 
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
LINES TERMINATED BY '\n'


