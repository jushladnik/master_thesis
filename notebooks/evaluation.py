import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import tqdm  
import ast
import gensim
from gensim.corpora import Dictionary

from top2vec import Top2Vec
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
import hdbscan
from sklearn.feature_extraction.text import CountVectorizer

import torch
from transformers import AutoModel, AutoTokenizer




import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ROOT_PATH = '/home/jhladnik/'


def get_top_words_for_topics_lda(trained_model, n_words):
    """ Returns top n words for each topic in trained model.
    :param trained_model: Trained LDA model.
    :param n_words: Number of top words to return.
    :return: List of lists of top words for each topic.
    """
    term_topic = trained_model.get_topics()

    topics_representations = []
    for topic in term_topic:
        top_k_words_ind = np.argsort(topic)[-n_words:]
        top_k_words = list(reversed([trained_model.id2word[i] for i in top_k_words_ind]))
        topics_representations.append(top_k_words)

    return topics_representations


def get_top_words_for_topics_bertopic(topic_model, n_words):
    """ Get top words for each topic from BERTopic model
    :param topic_model: trained BERTopic model 
    :param n_words: number of top words to return
    :return: list of lists of top words for each topic
    """
    topics = topic_model.get_topics()
    top_words = []
    for topic_id, topic in topics.items():
        if topic_id == 0:
            continue
        top_words.append([word for (word, score) in topic])
    return top_words

def get_top_words_for_topics_top2vec(trained_model, top_n_words, num_topics=15):
    topic_words, word_scores, topic_nums = trained_model.get_topics()
    num_identified_topics = len(topic_words)
    if num_identified_topics <= num_topics:
        topic_representations = topic_words
        print(f"identified {num_identified_topics} topics, but {num_topics} were requested")
    else:
        topic_mapping = trained_model.hierarchical_topic_reduction(num_topics=num_topics)
        topic_representations = trained_model.topic_words_reduced
    
    topic_representations = [topic_words[:top_n_words] for topic_words in topic_representations]
    return topic_representations

def topic_diversity(topic_top_words, top_k_words=25):
    """ Calculates topic diversity as percentage of unique words in the top k words of all topics.
    Args:
        topic_top_words (list): list of lists of top words in each topic
        top_k_words (int): number of top words to consider
    """
    if topic_top_words is None:
        return -1
    
    if top_k_words > len(topic_top_words[0]):
        print("top_k_words is larger than the number of words in the topic")
        return -1

    all_words = [word for topic in topic_top_words for word in topic[:top_k_words]]
    unique_words = set(all_words)
    return len(unique_words) / (top_k_words * len(topic_top_words))


def topic_coherence(df, trained_model, topic_top_words, top_k_words=10, coherence_metric='c_npmi'):	
    from gensim.models import CoherenceModel
    from gensim.corpora import Dictionary
    if topic_top_words is None:
        return -1
    
    if top_k_words > len(topic_top_words[0]):
        print("top_k_words is larger than the number of words in the topic")
        return -1

    coherence_model_lda = CoherenceModel(
        topics=topic_top_words,
        texts=list(df['lemmatized_text']),
        dictionary=Dictionary(list(df['lemmatized_text'])),
        coherence=coherence_metric,
        topn=top_k_words)

    return coherence_model_lda.get_coherence()


def get_models_topic_words(df, model_name, param_name, param_value, seed, verbose=True):
    df_it = df[(df.model_name==model_name) & (df[param_name]==param_value) & (df.seed==seed)].reset_index()
    #print(df_it.loc[0, "topic_representations"])

    # remove the array( and dtype='<U{i}') from the string (neccessary because of the way the array is stored in the csv in case tof top2vec)
    processed = df_it.loc[0, "topic_representations"].replace("array(", "")
    dtypes_list = []
    for i in range(10, 16): # maybe needed to adjust in some case
        dtypes_list.append(f" dtype='<U{i}')")
    for dtype in dtypes_list:
        processed = processed.replace(dtype+",", "").replace(dtype, "")

    topic_words = ast.literal_eval(processed)
    if verbose:
        for topic in topic_words:
            print(" ".join(topic))
    return topic_words

def jaccard_index(set1, set2):
    return len(set1.intersection(set2)) / len(set1.union(set2))

def cosine_similarity(a, b):
    eps = 1e-4
    if np.linalg.norm(a) < eps or np.linalg.norm(b) < eps:
        return 0
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def get_word2vec_embedding(word, model):
    """
    Returns word embeddings for a given word.
    :param word: word to get embeddings for
    :param model: word2vec model
    :return: word embeddings
    """
    try:
        embedding = model[word]
    except KeyError:
        embedding = np.zeros(model.vector_size)
    return embedding

def get_fasttext_embedding(word, model):
    return model.get_word_vector(word)

def get_sloberta_cls_embedding(text, model, tokenizer):
    token_ids = tokenizer.encode(text)
    token_ids = torch.tensor(token_ids).unsqueeze(0)
    output = model(token_ids)[0].squeeze()
    cls_out = output[0]
    return cls_out.detach().numpy()

def get_sloberta_mean_embedding(text, model, tokenizer):
    token_ids = tokenizer.encode(text)
    token_ids = torch.tensor(token_ids).unsqueeze(0)
    output = model(token_ids)[0].squeeze()
    mean_out = output[1:-1].mean(axis=0)
    return mean_out.detach().numpy()

def get_maximum_word_matching(words_1, words_2, embedding_name, embedding_function, fasttext_model = None, word2vec_model=None, sloberta_model=None, sloberta_tokenizer=None, verbose=False):
    from scipy.optimize import linear_sum_assignment


    weight_matrix = np.zeros((len(words_1), len(words_2)))
    for i, word1 in enumerate(words_1):
        for j, word2 in enumerate(words_2):
            if embedding_name == "sloberta":
                weight_matrix[i,j] = cosine_similarity(embedding_function(word1, sloberta_model, sloberta_tokenizer), 
                                                    embedding_function(word2, sloberta_model, sloberta_tokenizer))
            elif embedding_name == "fasttext":
                weight_matrix[i,j] = cosine_similarity(embedding_function(word1, fasttext_model), 
                                                    embedding_function(word2, fasttext_model))
            elif embedding_name == "word2vec":
                weight_matrix[i,j] = cosine_similarity(embedding_function(word1, word2vec_model), 
                                                        embedding_function(word2, word2vec_model))
            else:
                raise ValueError("Embedding name not recognized. Please use 'sloberta', 'fasttext' or 'word2vec'.")
            if verbose:
                print(f"word1: {word1}, word2: {word2}, weight: {weight_matrix[i,j]}")
    
    row_ind, col_ind = linear_sum_assignment(-weight_matrix)
    matching = sorted(
        [(words_1[i], words_2[j], weight_matrix[i, j]) for i, j in zip(row_ind, col_ind)],
        key=lambda x: x[2], reverse=True)

    return matching, np.mean([x[2] for x in matching])

def get_maximum_topic_matching(topic_words_1, topic_words_2, metric_name, embedding_function, fasttext_model = None, word2vec_model=None, sloberta_model=None, sloberta_tokenizer=None, verbose=False):
    from scipy.optimize import linear_sum_assignment


    weight_matrix = np.zeros((len(topic_words_1), len(topic_words_2)))
    for i, topic1 in enumerate(topic_words_1):
        for j, topic2 in enumerate(topic_words_2):
            if metric_name == "jaccard":
                ## with jaccard index we directly get the topic similarity
                ## using other semantic similarity metrics we would need to first get maximum word matching
                weight_matrix[i,j] = jaccard_index(set(topic1), set(topic2))
            elif metric_name == "sloberta":
                _, weight_matrix[i,j] = get_maximum_word_matching(topic1, topic2, embedding_name="sloberta", embedding_function=embedding_function, sloberta_model=sloberta_model, sloberta_tokenizer=sloberta_tokenizer)
            elif metric_name == "fasttext":
                _, weight_matrix[i,j] = get_maximum_word_matching(topic1, topic2, embedding_name="fasttext", embedding_function=embedding_function, fasttext_model=fasttext_model)
            elif metric_name == "word2vec":
                _, weight_matrix[i,j] = get_maximum_word_matching(topic1, topic2, embedding_name="word2vec", embedding_function=embedding_function, word2vec_model=word2vec_model)
            else:
                raise ValueError(f"Metric {metric_name} not supported")
            if verbose:
                print(f"topic1: {topic1}, topic2: {topic2}, weight: {weight_matrix[i,j]}")

    row_ind, col_ind = linear_sum_assignment(-weight_matrix)
    topic_matching = sorted(
        [(topic_words_1[i], topic_words_2[j], weight_matrix[i, j]) for i, j in zip(row_ind, col_ind)],
        key=lambda x: x[2], reverse=True)
    

    return topic_matching, np.mean([x[2] for x in topic_matching])


def visualise_model_param_eval(df_news, df_tweets, param_name):
    # Create a figure with 2 columns and 3 rows
    fig, axs = plt.subplots(3, 2, figsize=(10, 10))
    palette = ["r", "b", "y", "k"]

    # Plot data1 in the first column
    if df_news is not None:
        sns.lineplot(data=df_news, x=param_name, y='topic_diversity', hue="model_name", ax=axs[0][0], palette=palette)
        sns.lineplot(data=df_news, x=param_name, y='topic_coherence_npmi', hue="model_name", ax=axs[1][0], palette=palette)
        sns.lineplot(data=df_news, x=param_name, y='topic_coherence_umass', hue="model_name", ax=axs[2][0], palette=palette)

    # Plot data2 in the second column
    if df_tweets is not None:
        sns.lineplot(data=df_tweets, x=param_name, y='topic_diversity', hue="model_name", ax=axs[0][1], palette=palette)
        sns.lineplot(data=df_tweets, x=param_name, y='topic_coherence_npmi', hue="model_name", ax=axs[1][1], palette=palette)
        sns.lineplot(data=df_tweets, x=param_name, y='topic_coherence_umass', hue="model_name", ax=axs[2][1], palette=palette)

    # Show the plot
    plt.tight_layout()
    plt.show()



def evaluate_num_topics(df, num_samples, root_path, year, dataset):
    df = df.sample(num_samples, random_state=42)

    start = time.time()
    dictionary = Dictionary(list(df['lemmatized_text']))
    corpus = [dictionary.doc2bow(text) for text in list(df['lemmatized_text'])]

    def join_lemmas(list_lemmas):
        return ' '.join(list_lemmas)
    df['lemmatized_body'] = df['lemmatized_text'].apply(join_lemmas)
    # Make an index to word dictionary.
    temp = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token
    results_dict = {"model_name": [], "num_topics": [], "topic_representations": [],
    "topic_diversity": [], "topic_coherence_npmi": [], "topic_coherence_umass":[], "seed":[]}

    embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cuda')
    top_n_words = 10
    num_topics = 15
    num_topics_params = [5, 8, 10, 15, 20, 25, 30]
    seeds = [1, 2, 3]
    for num_topics in tqdm.tqdm(num_topics_params):
        for seed in seeds:
        
            print(f"num_topics: {num_topics}")
            
            
            ## top2vec
            print("top2vec")
            trained_model = Top2Vec(
                documents=list(df["lemmatized_body"]), 
                speed="learn", 
                workers=24
            )
            topics_representations = get_top_words_for_topics_top2vec(
                trained_model=trained_model, 
                top_n_words=top_n_words, 
                num_topics=num_topics
            )
            results_dict["model_name"].append("top2vec")
            results_dict["num_topics"].append(num_topics)
            results_dict["topic_diversity"].append(topic_diversity(topics_representations, top_n_words))
            results_dict["topic_coherence_npmi"].append(topic_coherence(df, trained_model, topics_representations, top_k_words=top_n_words, coherence_metric="c_npmi"))
            results_dict["topic_coherence_umass"].append(topic_coherence(df, trained_model, topics_representations, top_k_words=top_n_words, coherence_metric="u_mass"))
            results_dict["seed"].append(seed)
            results_dict["topic_representations"].append(topics_representations)
            
            ## NMF
            print("NMF")
            chunksize = 2000
            passes = 20
            iterations = 400
            eval_every = None  # Don't evaluate model perplexity, takes too much time.

            trained_model = gensim.models.Nmf(
                corpus=corpus,
                id2word=id2word,
                #chunksize=chunksize,
                #alpha='auto',
                #eta='auto',
                #iterations=iterations,
                num_topics=num_topics,
                passes=passes,
                eval_every=eval_every,
                random_state=seed
            )


            topics_representations = get_top_words_for_topics_lda(trained_model, top_n_words)
            results_dict["model_name"].append("NMF")
            results_dict["num_topics"].append(num_topics)
            results_dict["topic_diversity"].append(topic_diversity(topics_representations, top_n_words))
            results_dict["topic_coherence_npmi"].append(topic_coherence(df, trained_model, topics_representations, top_k_words=top_n_words, coherence_metric="c_npmi"))
            results_dict["topic_coherence_umass"].append(topic_coherence(df, trained_model, topics_representations, top_k_words=top_n_words, coherence_metric="u_mass"))
            results_dict["seed"].append(seed)
            results_dict["topic_representations"].append(topics_representations)


            ## LDA
            print("LDA")
            chunksize = 2000
            passes = 20
            iterations = 400
            trained_model = gensim.models.ldamodel.LdaModel(
                corpus=corpus,
                id2word=id2word,
                chunksize=chunksize,
                alpha='auto',
                eta='auto',
                iterations=iterations,
                num_topics=num_topics,
                passes=passes,
                eval_every=eval_every,
                random_state=seed
            )

            topics_representations = get_top_words_for_topics_lda(trained_model, top_n_words)
            results_dict["model_name"].append("LDA")
            results_dict["num_topics"].append(num_topics)
            results_dict["topic_diversity"].append(topic_diversity(topics_representations, top_n_words))
            results_dict["topic_coherence_npmi"].append(topic_coherence(df, trained_model, topics_representations, top_k_words=top_n_words, coherence_metric="c_npmi"))
            results_dict["topic_coherence_umass"].append(topic_coherence(df, trained_model, topics_representations, top_k_words=top_n_words, coherence_metric="u_mass"))
            results_dict["seed"].append(seed)
            results_dict["topic_representations"].append(topics_representations)




            ## BERTopic
            print("bertopic")
            docs = list(df.lemmatized_body)
            umap_model = UMAP(n_neighbors=15, n_components=5, 
                    min_dist=0.0, metric='cosine', random_state=seed)
            trained_model = BERTopic(
                embedding_model=embedding_model, 
                umap_model=umap_model,
                calculate_probabilities=True, 
                verbose=False, 
                nr_topics=num_topics, 
                top_n_words=top_n_words
            )
            topics, probs = trained_model.fit_transform(docs)
            topics_representations = get_top_words_for_topics_bertopic(trained_model, top_n_words)
            results_dict["model_name"].append("BERTopic")
            results_dict["num_topics"].append(num_topics)
            results_dict["topic_diversity"].append(topic_diversity(topics_representations, top_n_words))
            results_dict["topic_coherence_npmi"].append(topic_coherence(df, trained_model, topics_representations, top_k_words=top_n_words, coherence_metric="c_npmi"))
            results_dict["topic_coherence_umass"].append(topic_coherence(df, trained_model, topics_representations, top_k_words=top_n_words, coherence_metric="u_mass"))
            results_dict["seed"].append(seed)
            results_dict["topic_representations"].append(topics_representations)




    df_results = pd.DataFrame(results_dict)
    name = f"{root_path}/masters-thesis/notebooks/num_topics_models_{dataset}_{num_samples}_{year}.csv"
    df_results.to_csv(name, index=False)
    print(f"Saved results to {name}")
    print(f"evaluation took {(time.time() - start)/60} minutes, evaluated {len(seeds)} seeds and {len(num_topics_params)} num_topics_params")
    return df_results



def evaluate_top_n_words(df, num_samples, root_path, year, dataset):
    df = df.sample(num_samples, random_state=42)

    start = time.time()
    dictionary = Dictionary(list(df['lemmatized_text']))
    corpus = [dictionary.doc2bow(text) for text in list(df['lemmatized_text'])]

    def join_lemmas(list_lemmas):
        return ' '.join(list_lemmas)
    df['lemmatized_body'] = df['lemmatized_text'].apply(join_lemmas)
    # Make an index to word dictionary.
    temp = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token
    results_dict = {"model_name": [], "top_n_words": [], "topic_representations":[],
    "topic_diversity": [], "topic_coherence_npmi": [], "topic_coherence_umass":[], "seed":[]}

    embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cuda')
    #top_n_words = 10
    num_topics = 15
    #num_topics_params = [5, 8, 10, 15, 20, 25, 30]
    top_n_words_param = [3, 5, 8, 10, 15, 20, 25]
    seeds = [1, 2, 3]
    for top_n_words in tqdm.tqdm(top_n_words_param):
        for seed in seeds:
        
            print(f"top_n_words: {top_n_words}")
            print(f"dataset {dataset} with seed {seed}")

            
            ## top2vec
            print("top2vec")
            trained_model = Top2Vec(
                documents=list(df["lemmatized_body"]), 
                speed="learn", 
                workers=24
            )
            topics_representations = get_top_words_for_topics_top2vec(
                trained_model=trained_model, 
                top_n_words=top_n_words, 
                num_topics=num_topics
            )
            results_dict["model_name"].append("top2vec")
            results_dict["top_n_words"].append(top_n_words)
            results_dict["topic_diversity"].append(topic_diversity(topics_representations, top_n_words))
            results_dict["topic_coherence_npmi"].append(topic_coherence(df, trained_model, topics_representations, top_k_words=top_n_words, coherence_metric="c_npmi"))
            results_dict["topic_coherence_umass"].append(topic_coherence(df, trained_model, topics_representations, top_k_words=top_n_words, coherence_metric="u_mass"))
            results_dict["seed"].append(seed)
            results_dict["topic_representations"].append(topics_representations)
            
            ## NMF
            print("NMF")
            chunksize = 2000
            passes = 20
            iterations = 400
            eval_every = None  # Don't evaluate model perplexity, takes too much time.

            trained_model = gensim.models.Nmf(
                corpus=corpus,
                id2word=id2word,
                #chunksize=chunksize,
                #alpha='auto',
                #eta='auto',
                #iterations=iterations,
                num_topics=num_topics,
                passes=passes,
                eval_every=eval_every,
                random_state=seed
            )


            topics_representations = get_top_words_for_topics_lda(trained_model, top_n_words)
            results_dict["model_name"].append("NMF")
            results_dict["top_n_words"].append(top_n_words)
            results_dict["topic_diversity"].append(topic_diversity(topics_representations, top_n_words))
            results_dict["topic_coherence_npmi"].append(topic_coherence(df, trained_model, topics_representations, top_k_words=top_n_words, coherence_metric="c_npmi"))
            results_dict["topic_coherence_umass"].append(topic_coherence(df, trained_model, topics_representations, top_k_words=top_n_words, coherence_metric="u_mass"))
            results_dict["seed"].append(seed)
            results_dict["topic_representations"].append(topics_representations)

            ## LDA
            print("LDA")
            chunksize = 2000
            passes = 20
            iterations = 400
            trained_model = gensim.models.ldamodel.LdaModel(
                corpus=corpus,
                id2word=id2word,
                chunksize=chunksize,
                alpha='auto',
                eta='auto',
                iterations=iterations,
                num_topics=num_topics,
                passes=passes,
                eval_every=eval_every,
                random_state=seed
            )

            topics_representations = get_top_words_for_topics_lda(trained_model, top_n_words)
            results_dict["model_name"].append("LDA")
            results_dict["top_n_words"].append(top_n_words)
            results_dict["topic_diversity"].append(topic_diversity(topics_representations, top_n_words))
            results_dict["topic_coherence_npmi"].append(topic_coherence(df, trained_model, topics_representations, top_k_words=top_n_words, coherence_metric="c_npmi"))
            results_dict["topic_coherence_umass"].append(topic_coherence(df, trained_model, topics_representations, top_k_words=top_n_words, coherence_metric="u_mass"))
            results_dict["seed"].append(seed)
            results_dict["topic_representations"].append(topics_representations)



            ## BERTopic
            print("bertopic")
            docs = list(df.lemmatized_body)
            umap_model = UMAP(n_neighbors=15, n_components=5, 
                    min_dist=0.0, metric='cosine', random_state=seed)
            trained_model = BERTopic(
                embedding_model=embedding_model, 
                umap_model=umap_model,
                calculate_probabilities=True, 
                verbose=False, 
                nr_topics=num_topics, 
                top_n_words=top_n_words
            )
            topics, probs = trained_model.fit_transform(docs)
            topics_representations = get_top_words_for_topics_bertopic(trained_model, top_n_words)
            results_dict["model_name"].append("BERTopic")
            results_dict["top_n_words"].append(top_n_words)
            results_dict["topic_diversity"].append(topic_diversity(topics_representations, top_n_words))
            results_dict["topic_coherence_npmi"].append(topic_coherence(df, trained_model, topics_representations, top_k_words=top_n_words, coherence_metric="c_npmi"))
            results_dict["topic_coherence_umass"].append(topic_coherence(df, trained_model, topics_representations, top_k_words=top_n_words, coherence_metric="u_mass"))
            results_dict["seed"].append(seed)
            results_dict["topic_representations"].append(topics_representations)




    df_results = pd.DataFrame(results_dict)
    name = f"{root_path}/masters-thesis/notebooks/top_n_words_models_{dataset}_{num_samples}_{year}.csv"
    df_results.to_csv(name, index=False)
    print(f"Saved results to {name}")
    print(f"evaluation took {(time.time() - start)/60} minutes, evaluated {len(seeds)} seeds and {len(top_n_words_param)} top_n_words_param")
    return df_results





def evaluate_num_documents(df, num_samples, root_path, dataset, sample_documents):
   
    df_all = df
    start = time.time()
    dictionary = Dictionary(list(df['lemmatized_text']))
    corpus = [dictionary.doc2bow(text) for text in list(df['lemmatized_text'])]

    def join_lemmas(list_lemmas):
        return ' '.join(list_lemmas)
    df['lemmatized_body'] = df['lemmatized_text'].apply(join_lemmas)
    # Make an index to word dictionary.
    temp = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token
    results_dict = {"model_name": [], "num_documents": [], "topic_representations": [],
    "topic_diversity": [], "topic_coherence_npmi": [], "topic_coherence_umass":[], "seed":[]}

    embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cuda')
    top_n_words = 10
    num_topics = 15
    min_topic_size = 100
    #num_topics_params = [5, 8, 10, 15, 20, 25, 30]
    #top_n_words_param = [5, 8, 10, 15, 20, 25]
    num_documents_params = [5000, 10000, 20000, 35000, 50000, 65000, 80000, 95000]
    seeds = [1, 2, 3]
    for num_documents in tqdm.tqdm(num_documents_params):
        
        for seed in seeds:
            if sample_documents:
                # select random documents that dont overlap between seeds
                #df = df_all.sample(n=num_documents, random_state=seed)
                df = df_all.sample(frac=1, random_state=1).reset_index(drop=True)
                df = df.iloc[num_documents*(seed-1):num_documents*seed]
            else:
                # always select the same documents, randomization happens at the model level
                df = df_all.sample(frac=1, random_state=1).reset_index(drop=True)
                df = df.iloc[0:num_documents]
        
            print(f"num_documents: {num_documents}")
            print(f"dataset {dataset} with seed {seed}")
            
            
            try:
                ## top2vec
                print("top2vec")
                trained_model = Top2Vec(
                    documents=list(df["lemmatized_body"]), 
                    speed="learn", 
                    workers=24
                )
                topics_representations = get_top_words_for_topics_top2vec(
                    trained_model=trained_model, 
                    top_n_words=top_n_words, 
                    num_topics=num_topics
                )
                results_dict["model_name"].append("top2vec")
                results_dict["num_documents"].append(num_documents)
                results_dict["topic_diversity"].append(topic_diversity(topics_representations, top_n_words))
                results_dict["topic_coherence_npmi"].append(topic_coherence(df, trained_model, topics_representations, top_k_words=top_n_words, coherence_metric="c_npmi"))
                results_dict["topic_coherence_umass"].append(topic_coherence(df, trained_model, topics_representations, top_k_words=top_n_words, coherence_metric="u_mass"))
                results_dict["seed"].append(seed)
                results_dict["topic_representations"].append(topics_representations)
            except ValueError as e:
                print(f"top2vec failed with error {e} for num_documents {num_documents} and seed {seed}")
            
            ## NMF
            print("NMF")
            chunksize = 2000
            passes = 20
            iterations = 400
            eval_every = None  # Don't evaluate model perplexity, takes too much time.

            trained_model = gensim.models.Nmf(
                corpus=corpus,
                id2word=id2word,
                #chunksize=chunksize,
                #alpha='auto',
                #eta='auto',
                #iterations=iterations,
                num_topics=num_topics,
                passes=passes,
                eval_every=eval_every,
                random_state=seed
            )


            topics_representations = get_top_words_for_topics_lda(trained_model, top_n_words)
            results_dict["model_name"].append("NMF")
            results_dict["num_documents"].append(num_documents)
            results_dict["topic_diversity"].append(topic_diversity(topics_representations, top_n_words))
            results_dict["topic_coherence_npmi"].append(topic_coherence(df, trained_model, topics_representations, top_k_words=top_n_words, coherence_metric="c_npmi"))
            results_dict["topic_coherence_umass"].append(topic_coherence(df, trained_model, topics_representations, top_k_words=top_n_words, coherence_metric="u_mass"))
            results_dict["seed"].append(seed)
            results_dict["topic_representations"].append(topics_representations)

            ## LDA
            print("LDA")
            chunksize = 2000
            passes = 20
            iterations = 400
            trained_model = gensim.models.ldamodel.LdaModel(
                corpus=corpus,
                id2word=id2word,
                chunksize=chunksize,
                alpha='auto',
                eta='auto',
                iterations=iterations,
                num_topics=num_topics,
                passes=passes,
                eval_every=eval_every,
                random_state=seed
            )

            topics_representations = get_top_words_for_topics_lda(trained_model, top_n_words)
            results_dict["model_name"].append("LDA")
            results_dict["num_documents"].append(num_documents)
            results_dict["topic_diversity"].append(topic_diversity(topics_representations, top_n_words))
            results_dict["topic_coherence_npmi"].append(topic_coherence(df, trained_model, topics_representations, top_k_words=top_n_words, coherence_metric="c_npmi"))
            results_dict["topic_coherence_umass"].append(topic_coherence(df, trained_model, topics_representations, top_k_words=top_n_words, coherence_metric="u_mass"))
            results_dict["seed"].append(seed)
            results_dict["topic_representations"].append(topics_representations)



            ## BERTopic
            print("bertopic")
            docs = list(df.lemmatized_body)
            umap_model = UMAP(n_neighbors=15, n_components=5, 
                    min_dist=0.0, metric='cosine', random_state=seed)

            trained_model = BERTopic(
                embedding_model=embedding_model, 
                umap_model=umap_model,
                #vectorizer_model=vectorizer_model,
                calculate_probabilities=True, 
                verbose=False, 
                min_topic_size=min_topic_size,
                nr_topics=num_topics, 
                top_n_words=top_n_words
            )
            topics, probs = trained_model.fit_transform(docs)
            topics_representations = get_top_words_for_topics_bertopic(trained_model, top_n_words)
            results_dict["model_name"].append("BERTopic")
            results_dict["num_documents"].append(num_documents)
            results_dict["topic_diversity"].append(topic_diversity(topics_representations, top_n_words))
            results_dict["topic_coherence_npmi"].append(topic_coherence(df, trained_model, topics_representations, top_k_words=top_n_words, coherence_metric="c_npmi"))
            results_dict["topic_coherence_umass"].append(topic_coherence(df, trained_model, topics_representations, top_k_words=top_n_words, coherence_metric="u_mass"))
            results_dict["seed"].append(seed)
            results_dict["topic_representations"].append(topics_representations)




    df_results = pd.DataFrame(results_dict)
    if sample_documents:
        name = f"{root_path}/masters-thesis/notebooks/num_documents_models_{dataset}_{num_samples}_sampled_docs.csv"
    else:
        name = f"{root_path}/masters-thesis/notebooks/num_documents_models_{dataset}_{num_samples}.csv"
    df_results.to_csv(name, index=False)
    print(f"Saved results to {name}")
    print(f"evaluation took {(time.time() - start)/60} minutes, evaluated {len(seeds)} seeds and {len(num_documents_params)} num_documents_params")
    return df_results




def evaluate_bertopic_param(df, num_samples, root_path, dataset, eval_param_name, evaluation_param_values, seeds):

    dictionary = Dictionary(list(df['lemmatized_text']))
    corpus = [dictionary.doc2bow(text) for text in list(df['lemmatized_text'])]

    def join_lemmas(list_lemmas):
        return ' '.join(list_lemmas)
    df['lemmatized_body'] = df['lemmatized_text'].apply(join_lemmas)

    results_dict = {"model_name": [], eval_param_name: [], "topic_representations": [],
    "topic_diversity": [], "topic_coherence_npmi": [], "topic_coherence_umass":[], "seed":[]}

    embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cuda')

    ## default parameter values
    top_n_words = 10
    num_topics = 15
    min_topic_size = 10
    min_df = 10
    n_gram_range = (1, 1)
    n_neighbors = 15
    
    for param_value in tqdm.tqdm(evaluation_param_values):
        for seed in seeds:
            print(f"{eval_param_name}: {param_value}")
            print(f"dataset {dataset} with seed {seed}")

            if eval_param_name == "min_topic_size":
                min_topic_size = param_value
            elif eval_param_name == "min_df":
                min_df = param_value
            elif eval_param_name == "top_n_words":
                top_n_words = param_value
            elif eval_param_name == "num_topics":
                num_topics = param_value
            elif eval_param_name == "n_gram_range":
                n_gram_range = param_value
            elif eval_param_name == "n_neighbors":
                n_neighbors = param_value
            else:
                raise ValueError(f"eval_param_name not found with value: {eval_param_name}")
            

            
            ## BERTopic
            print("bertopic")
            docs = list(df.lemmatized_body)
            vectorizer_model = CountVectorizer(
                #min_df=min_df,
                ngram_range=n_gram_range
            )

            umap_model = UMAP(n_neighbors=n_neighbors, n_components=5, 
                    min_dist=0.0, metric='cosine', random_state=seed)
            

            trained_model = BERTopic(
                embedding_model=embedding_model, 
                umap_model=umap_model,
                vectorizer_model=vectorizer_model,
                calculate_probabilities=True, 
                verbose=False, 
                min_topic_size=min_topic_size,
                nr_topics=num_topics, 
                top_n_words=top_n_words
            )
            topics, probs = trained_model.fit_transform(docs)
            topics_representations = get_top_words_for_topics_bertopic(trained_model, top_n_words)
            results_dict["model_name"].append("BERTopic")
            results_dict[eval_param_name].append(param_value)
            results_dict["topic_diversity"].append(topic_diversity(topics_representations, top_n_words))
            results_dict["topic_coherence_npmi"].append(topic_coherence(df, trained_model, topics_representations, top_k_words=top_n_words, coherence_metric="c_npmi"))
            results_dict["topic_coherence_umass"].append(topic_coherence(df, trained_model, topics_representations, top_k_words=top_n_words, coherence_metric="u_mass"))
            results_dict["seed"].append(seed)
            results_dict["topic_representations"].append(topics_representations)



    df_results = pd.DataFrame(results_dict)
    name = f"{root_path}/masters-thesis/notebooks/{eval_param_name}_results_{dataset}_{num_samples}.csv"
    df_results.to_csv(name, index=False)
    print(f"Saved results to {name}")
    #print(f"evaluation took {(time.time() - start)/60} minutes, evaluated {len(seeds)} seeds and {len(evaluation_param_values)} min_df_params")
    return df_results



def test_evaluate_bertopic_param():
    start = time.time()
    ROOT_PATH = '/home/jhladnik/'
    eval_param_name="min_topic_size" 
    evaluation_param_values=  [15, 20, 35, 50,  80, 100, 200, 500] #[(1,1), (1,2), (1,3)] #[10, 20, 50, 100] #[5, 10, 15, 20, 25, 50]  #[5, 10, 15, 20, 25], 
    seeds=[1, 2, 3]  #[1, 2, 3]

    num_samples_params = [5000, 50000]
    for num_samples in num_samples_params:
        df_news = pd.read_parquet(
            f'{ROOT_PATH}/data/eventregistry/df_news_lemmas_{num_samples}.parquet.gzip')
        
        df_tweets = pd.read_parquet(
            f'{ROOT_PATH}/data/sl-tweets/df_tweets_lemmas_{num_samples}.parquet.gzip')

        ## evaluate datasets
        datasets = {"news":df_news, "tweets":df_tweets}
        for dataset_name in datasets:
            df = datasets[dataset_name]
            evaluate_bertopic_param(
                df=df, 
                num_samples=num_samples, 
                root_path=ROOT_PATH, 
                dataset=dataset_name, 
                eval_param_name=eval_param_name, 
                evaluation_param_values=evaluation_param_values, 
                seeds=seeds, 
                )

    print(f"evaluation of {eval_param_name} with values: {evaluation_param_values} and seeds: {seeds} took {(time.time() - start)/60} minutes")


def evaluation_of_num_documents_variance():
    
    NUM_SAMPLES = 300_000
    year = 2020
    ROOT_PATH = '/home/jhladnik/'

    start = time.time()

    df = pd.read_parquet(
        f'{ROOT_PATH}/data/sl-tweets/df_tweets_{year}_lemmas_{NUM_SAMPLES}.parquet.gzip')
    evaluate_num_documents(df, num_samples=NUM_SAMPLES, root_path=ROOT_PATH, dataset="tweets", sample_documents=True)
    evaluate_num_documents(df, num_samples=NUM_SAMPLES, root_path=ROOT_PATH, dataset="tweets", sample_documents=False)

    

    df = pd.read_parquet(
        f'{ROOT_PATH}/data/eventregistry/df_news_{year}_lemmas_{NUM_SAMPLES}.parquet.gzip')
    evaluate_num_documents(df, num_samples=NUM_SAMPLES, root_path=ROOT_PATH, dataset="news", sample_documents=True)
    evaluate_num_documents(df, num_samples=NUM_SAMPLES, root_path=ROOT_PATH, dataset="news", sample_documents=False)

    print(f"evaluation of both news and tweets num documents took {(time.time() - start)/60} minutes")



if __name__ == '__main__':
    #evaluation_of_num_documents_variance()

    #test_evaluate_bertopic_param()
    
    
    
    dataset_samples = 300_000
    num_samples = 50_000
    ROOT_PATH = '/home/jhladnik/'
    year = 2020

    ## BEWARE OF DATASET NAMES
    df = pd.read_parquet(
        f'{ROOT_PATH}/data/sl-tweets/df_tweets_{year}_lemmas_{dataset_samples}.parquet.gzip')
    evaluate_num_topics(df, num_samples=num_samples, root_path=ROOT_PATH, year=year, dataset="tweets")
    #evaluate_top_n_words(df, num_samples=num_samples, root_path=ROOT_PATH, year=year, dataset="tweets")

    df = pd.read_parquet(
        f'{ROOT_PATH}/data/eventregistry/df_news_{year}_lemmas_{dataset_samples}.parquet.gzip')
    evaluate_num_topics(df, num_samples=num_samples, root_path=ROOT_PATH, year=year, dataset="news")
    #evaluate_top_n_words(df, num_samples=num_samples, root_path=ROOT_PATH, year=year, dataset="news")
    
    
