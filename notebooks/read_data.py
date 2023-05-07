import json
import pandas as pd
import numpy as np
import os
import gzip
import time
import tqdm

from datetime import datetime

# test for git push


def read_json_file(filepath):
    """
    Reads a json file.
    :param filepath: path to a file to read
    :return data: json data from file  
    """

    with open(filepath, encoding="utf-8") as infile:
        data = json.load(infile)

    return data


def read_news_raw_data_json(filepath):
    """
    This function takes a single json file that is a page of results from a single year.
    Returns list of dictionaries that contain info (that we are interested in) about single articles. 
    Each dictionary is a single article.

    :param filepath: path to json file
    :return article_dicts:  (new articles added)
    """

    json_data = read_json_file(filepath)
    # a dictionary (JSON) of all articles' metadata
    articles_full = json_data['articles']['results']

    # can later add more fields if necessary (such as url,...)
    article_dicts = []
    for article in articles_full:
        article_dict = {}
        article_dict['body'] = article['body']
        article_dict['media'] = article['source']['title']
        article_dict['title'] = article['title']
        article_dict['date'] = article['date']

        article_dicts.append(article_dict)

    return article_dicts


def read_year_news_raw_into_df(path, year):
    """
    Creates a dataframe of news articles for a specified year.
    :param path: path to the news data directory
    :param year: year we want to make dataframe from
    :return: dataframe with data about news from specified year
    """
    year_path = os.path.join(path, year)
    df_year = pd.DataFrame()
    for filename in tqdm.tqdm(os.listdir(year_path)):
        # print(filename)

        f = os.path.join(year_path, filename)
        # checking if it is a file
        try:
            articles = read_news_raw_data_json(f)
        except ValueError as e:
            print('invalid json: %s' % e)
            return None  # or: raise

        df = pd.DataFrame(articles)
        df_year = pd.concat([df_year, df], ignore_index=True)
        # print(df_year)
    return df_year


def read_tweets_raw_data(filepath):
    """
    This function takes a single filepath of json.gz file containing tweets.
    Returns list of dictionaries that contain info (that we are interested in) about single tweets. 
    Each dictionary is a single article.

    :param filepath: dictionary (JSON)
    :return article_dicts:  (new articles added)
    """

    with gzip.open(filepath, 'r') as fin:
        data = json.loads(fin.read().decode('utf-8'))

    tweets_full = data       # a dictionary (JSON) of all articles' metadata

    # can later add more fields if necessary (such as url,...)
    tweet_dicts = []
    for tweet in tweets_full:
        tweet_dict = {}
        tweet_dict['full_text'] = tweet['full_text']
        tweet_dict['created_at'] = tweet['created_at']
        tweet_dict['user_screen_name'] = tweet['user']['screen_name']
        tweet_dict['in_reply_to_user_id'] = tweet['in_reply_to_user_id']
        tweet_dict['is_quote_status'] = tweet['is_quote_status']
        tweet_dict['user_id'] = tweet['user']['id']
        tweet_dict['id'] = tweet['id']

        tweet_dicts.append(tweet_dict)

    return tweet_dicts


def read_tweets_raw_into_df(path):
    """
    Creates a dataframe of tweets .
    :param path: path to the tweet data directory
    :return: dataframe with data about tweets
    """
    df_full = pd.DataFrame()
    files = os.listdir(path)
    for filename in tqdm.tqdm(files):
        # print(filename)
        if not filename.endswith('.gz'):
            continue

        f = os.path.join(path, filename)
        # checking if it is a file
        try:
            tweets = read_tweets_raw_data(f)
        except ValueError as e:
            print('invalid json: %s' % e)
            return None  # or: raise

        df = pd.DataFrame(tweets)
        df_full = pd.concat([df_full, df], ignore_index=True)
    return df_full


def read_jsonl_file(filepath):
    """
    Reads a jsonl file.
    :param filepath: path to a file to read
    :return data: liston of json objects about articles 
    """

    with open(filepath, 'r') as json_file:
        json_list = list(json_file)

    articles_json = []
    for json_str in json_list:
        result = json.loads(json_str)
        articles_json.append(result)
    return articles_json


def read_news2021_22_raw_data_jsonl(filepath):
    """
    This function takes a single json file that is a day of results from a single year.
    Returns list of dictionaries that contain info (that we are interested in) about single articles. 
    Each dictionary is a single article.

    :param filepath: path to json file
    :return article_dicts:  (new articles added)
    """

    articles_json = read_jsonl_file(filepath)

    # can later add more fields if necessary (such as url,...)
    article_dicts = []
    for article in articles_json:
        article_dict = {}
        article_dict['body'] = article['body']
        article_dict['media'] = article['source']['title']
        article_dict['title'] = article['title']
        article_dict['date'] = article['date']

        article_dicts.append(article_dict)

    return article_dicts


def read_2021_22_year_news_raw_into_df(path, year):
    """
    Creates a dataframe of news articles for a specified year.
    :param path: path to the news data directory
    :param year: year we want to make dataframe from
    :return: dataframe with data about news from specified year
    """
    year_path = os.path.join(path, year)
    df_year = pd.DataFrame()
    for filename in tqdm.tqdm(os.listdir(year_path)):
        # print(filename)

        f = os.path.join(year_path, filename)
        # checking if it is a file
        try:
            articles = read_news2021_22_raw_data_jsonl(f)
        except ValueError as e:
            print('invalid json: %s' % e)
            return None  # or: raise

        df = pd.DataFrame(articles)
        df_year = pd.concat([df_year, df], ignore_index=True)
        # print(df_year)
    return df_year


def read_tweets_into_yearly_dfs():

    path = "/home/jhladnik/data/sl-tweets/sl_tweets_index_dump.jsonl.gz"
    tweet_dicts = []
    start = time.time()
    count = 0
    with gzip.open(path,'rt') as f:
        for line in tqdm.tqdm(f):
            data = json.loads(line)
            #print(data['text'].encode().decode())
            #print()
            count+=1
            if data['language_inferred'] != "sl":
                continue
            
            tweet_dict = {}
            tweet_dict['full_text'] = data['text']
            tweet_dict['created_at'] = data['pub_date']
            tweet_dict["year"] = datetime.strptime(data["pub_date"][:10], "%Y-%m-%d").year
            tweet_dict['user_screen_name'] = data['username']
            tweet_dict['in_reply_to_screen_name'] = data['in_reply_to_screen_name']
            tweet_dict['quoted_status_id_str'] = data['quoted_status_id_str']

            tweet_dicts.append(tweet_dict)
            
            #print( datetime.strptime(data["pub_date"][:10], "%Y-%m-%d").year)
            
            #print('got line', line)
            if count % 1000000 == 0:
                print(f"read {count/1000000} mio tweets in {time.time()-start} seconds")
            

    df_all_tweets = pd.DataFrame(tweet_dicts)

    for year in [2017, 2018, 2019, 2020, 2021]:
        df_year_tweets = df_all_tweets[df_all_tweets["year"] == year]
        df_year_tweets.to_parquet(f"/home/jhladnik/data/sl-tweets/df_tweets_{year}.parquet.gzip", compression='gzip')


if __name__ == "__main__":
    """     start = time.time()
    filepath = os.path.join(EVENTREGISTRY_PATH, "2020", "clanki.2020.page3.json")
    df_year_news =  read_year_news_raw_into_df(EVENTREGISTRY_PATH, '2020')
    df_year_news.to_parquet('C:/Users/hladn/FAKS/Magistrsko delo/data/eventregistry/df_news_2020.parquet.gzip',compression='gzip')
    df_year_news
    end = time.time()
    print(f"reading news data into dataframe for year 2020 took {end - start} seconds")
    """

    """
    year = 2021
    start = time.time()
    TWEETS_PATH = "/home/jhladnik/data/sl-tweets"
    all_tweets = read_tweets_raw_into_df(f'{TWEETS_PATH}/sl-tweets-{year}')
    all_tweets.to_parquet(
        f'/home/jhladnik/data/sl-tweets/df_sl_tweets_{year}.parquet.gzip', compression='gzip')
    end = time.time()
    print(
        f"reading tweets data into dataframe and saving to parquet for year {year} took {end - start} seconds")
    """

    read_tweets_into_yearly_dfs()
