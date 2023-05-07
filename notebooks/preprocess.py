import time
import pandas as pd
import numpy as np

import nltk
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.manifold import TSNE
# from sklearn.manifold import MDS
# from sklearn.decomposition import PCA
import string
import re
from collections import Counter
from gensim.utils import simple_preprocess
import gensim
from utils import classla_lemmatize, preprocess_text, preprocess_and_lemmatize_news, preprocess_and_lemmatize_tweets
SEED = 42
ROOT_PATH = '/home/jhladnik/'

def preprocess_news(NUM_SAMPLES, year):
    # 450k news articles
    df_all = pd.read_parquet(
    f"{ROOT_PATH}data/eventregistry/df_news_{year}.parquet.gzip")

    df = df_all.sample(NUM_SAMPLES, random_state=SEED)

    start = time.time()
    df = preprocess_and_lemmatize_news(df)
    # save preprocessed df
    df.to_parquet(
        f'{ROOT_PATH}/data/eventregistry/df_news_{year}_lemmas_{NUM_SAMPLES}.parquet.gzip', compression='gzip')


    print(f"Preprocessing {NUM_SAMPLES} news of year {year} took {(time.time() - start)/60} minutes")


def preprocess_tweets(NUM_SAMPLES, year):
    # 12mio tweets
    ROOT_PATH = "/home/jhladnik"
    df_all = pd.read_parquet(
        f'{ROOT_PATH}/data/sl-tweets/df_tweets_{year}.parquet.gzip')

    df = df_all[df_all.in_reply_to_screen_name.isna()].sample(NUM_SAMPLES, random_state=SEED)
    start = time.time()
    df = preprocess_and_lemmatize_tweets(df)
    # save preprocessed df
    df.to_parquet(
        f'{ROOT_PATH}/data/sl-tweets/df_tweets_{year}_lemmas_{NUM_SAMPLES}.parquet.gzip', compression='gzip')  

    print(f"Preprocessing {NUM_SAMPLES} tweets for year {year} took {(time.time() - start)/60} minutes")

def preprocess_tweets_21_old(NUM_SAMPLES):
    # 12mio tweets
    ROOT_PATH = "/home/jhladnik"
    df_all = pd.read_parquet(
        f'{ROOT_PATH}/data/sl-tweets/df_sl_tweets_21.parquet.gzip')

    df = df_all.sample(NUM_SAMPLES, random_state=SEED)
    start = time.time()
    df = preprocess_and_lemmatize_tweets(df)
    # save preprocessed df
    df.to_parquet(
        f'{ROOT_PATH}/data/sl-tweets/df_tweets_lemmas_{NUM_SAMPLES}.parquet.gzip', compression='gzip')  

    print(f"Preprocessing of {NUM_SAMPLES} for old 2021 tweets took {(time.time() - start)/60} minutes")

if __name__ == "__main__":
    
    NUM_SAMPLES = 300_000
    year = "2021"

    #preprocess_news(NUM_SAMPLES, year)
    #preprocess_tweets_21_old(NUM_SAMPLES)

    for year in ["2019", "2020", "2021"]:
        preprocess_tweets(NUM_SAMPLES, year)
    #preprocess_tweets(NUM_SAMPLES, year)
