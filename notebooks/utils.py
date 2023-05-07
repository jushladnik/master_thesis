from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

import string
import re

import pandas as pd


from gensim.utils import simple_preprocess
import classla
ROOT_PATH = '/home/jhladnik/'




#classla.download('sl')        # download non-standard models for Slovenian, use hr for Croatian and sr for Serbian
#classla.download('sl', type='nonstandard')        # download non-standard models for Slovenian, use hr for Croatian and sr for Serbian

#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')


def get_all_sl_stopwords():
    # get all slovene stopwords
    # return: list of stopwords
    filepath = f"{ROOT_PATH}data/stopwords.txt"
    with open(filepath, 'r', encoding='utf-8') as f:
        additional_stopwords = f.read().splitlines()

    filepath = f"{ROOT_PATH}data/stopwords_jus.txt"
    with open(filepath, 'r', encoding='utf-8') as f:
        jus_stopwords = f.read().splitlines()

    stop_words = stopwords.words(
        'slovene') + list(string.punctuation) + additional_stopwords + jus_stopwords
    new_sw = ["rt", "href", "http", "https", "quot", "nbsp", "mailto", "mail", "getty", "foto", "images", "urbanec",
              "sportid"]
    stop_words.extend(new_sw)
    return stop_words


def print_some_texts(columns, df):
    # print some texts
    # columns: list of columns to print
    # df: dataframe
    # 7240, 7241, 8013, 14500, 16500, 16304, 18300,  21750, 34036]
    text_idxs = [1, 2, 3]
    for i in text_idxs:
        for column in columns:
            print(df[column].iloc[i])


def preprocess_text(text):
    """Preprocess text for topic modeling. remove any urls, emails, twitter handles, new lines, special characters, tags, punctuations, and convert to lowercase. retain stopwords
    :param text: text to preprocess
    :return: preprocessed text
    """
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"http\S+", " ", text)  # remove urls
    text = re.compile('\S*@\S*\s?').sub(r'', text)  # remove mails
    text = re.sub("@[A-Za-z0-9]+", "", text)  # remove twitter handle
    text = re.sub('\s+', ' ', text)  # remove new line
    # &amp; is a special character for ampersand
    text = re.sub("&amp;", "", text)
    text = re.sub('<USER>', '',
                  text)  # remove '<USER>' as there are some such strings as user or url is masked with this string
    text = re.sub('<URL>', '', text)
    text = re.sub('[^a-zA-ZčšžČŠŽđĐćĆ]', ' ', text)  # Remove punctuations
    text = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)  # remove tags
    # remove special characters and digits
    text = re.sub("(\\d|\\W)+", " ", text)
    return text


def tokenize(text):
    """Tokenize text for topic modeling. Use NLTK's word_tokenize and remove stopwords and words shorter than 3 characters.
    :param text: text to tokenize
    :return: tokenized text"""
    text = preprocess_text(text)
    tokens = word_tokenize(text)
    stop_words = get_all_sl_stopwords()
    filtered_tokens = []
    # Filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation). (adapted from lab example)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            if token not in stop_words and len(token) > 2:
                filtered_tokens.append(token)
    return filtered_tokens


def classla_lemmatize(text, nlp_pipeline):
    """Lemmatize text for topic modeling. Use classla's lemmatizer and remove stopwords and words shorter than 3 characters.
    :param text: text to lemmatize
    :return: lemmatized text"""

    stop_words = get_all_sl_stopwords()

    try:
        doc = nlp_pipeline(text)
        lemmas = [word.lemma for sent in doc.sentences for word in sent.words]

        preprocessed_lemmas = []  # a list of words of a single article
        for lemma in lemmas:
            # remove all words shorter than three characters
            lemma_len = len(lemma)
            if (lemma not in stop_words) and lemma_len >= 3 and lemma_len <= 25:
                preprocessed_lemmas.append(lemma)

        # in case all words were stopwords we add a placeholder. we later remove it when we drop texts (rows) with less than n lemmas
        if len(preprocessed_lemmas) == 0:
            preprocessed_lemmas = ['a']
    except Exception as e:
        print(f"exception was raised at lemmatizingtext: {text} with exception:{e}")
        preprocessed_lemmas = ['a']
    return preprocessed_lemmas


def preprocess_and_lemmatize_news(df):
    """Preprocess and lemmatize news text.
    :param df: dataframe with news text
    :return: dataframe with preprocessed and lemmatized news text as new column
    """
    # create a new column for preprocessed and lemmatized text
    config = {
        # Comma-separated list of processors to use
        'processors': 'tokenize, pos, lemma',
        'lang': 'sl',  # Language code for the language to build the Pipeline in
        # Use pretokenized text as input and disable tokenization
        'tokenize_pretokenized': False,
        'use_gpu': True,
        # initialize the default non-standard Slovenian pipeline, use hr for Croatian and sr for Serbian
        'type': 'nonstandard'
    }
    nlp_pipeline = classla.Pipeline(**config)
    df['preprocessed_text'] = df['body'].apply(preprocess_text)
    df['lemmatized_text'] = df['preprocessed_text'].apply(lambda x: classla_lemmatize(x, nlp_pipeline))
    return df




def preprocess_and_lemmatize_tweets(df):
    """Preprocess and lemmatize news text.
    :param df: dataframe with news text
    :return: dataframe with preprocessed and lemmatized news text as new column
    """
    # create a new column for preprocessed and lemmatized text
    config = {
        # Comma-separated list of processors to use
        'processors': 'tokenize, pos, lemma',
        'lang': 'sl',  # Language code for the language to build the Pipeline in
        # Use pretokenized text as input and disable tokenization
        'tokenize_pretokenized': False,
        'use_gpu': True,
        # initialize the default non-standard Slovenian pipeline, use hr for Croatian and sr for Serbian
        'type': 'nonstandard'
    }
    nlp_pipeline = classla.Pipeline(**config)
    df['preprocessed_text']=df['full_text'].apply(preprocess_text)
    #df['tokenized_text']= df['preprocessed_text'].apply(tokenize)
    df[df.preprocessed_text.str.len() > 0]
    df['lemmatized_text'] = df['preprocessed_text'].apply(lambda x: classla_lemmatize(x, nlp_pipeline))
    return df



