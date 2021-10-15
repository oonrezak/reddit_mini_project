import string
import re
from collections import Counter
from collections import defaultdict

import matplotlib as mpl
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
import seaborn as sns
sns.set_context("talk")

mpl.rcParams['lines.markersize'] = 15

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
from sklearn.metrics import calinski_harabasz_score, silhouette_score

from IPython.display import HTML

from PIL import Image as PILImage

from wordcloud import WordCloud

def clean(df):
    """
    Cleans the post titles in the DataFrame by removing punctuation
    marks, and formatting all letters to lowercase.

    Parameter
    ---------
    df : pd.DataFrame
        the raw DataFrame

    Returns
    -------
    df : pd.DataFrame
        the DataFrame with an extra column, `clean_title`, that contains the 
        'clean' post titles
        
    """
    # Adding 'curly' quotes to string.punctuation so they are removed:
    puncts = string.punctuation + "‘’“”"
    
    # Defining another function to clean the titles:
    def clean_title(raw_title):
        """
        Cleans each individual post title. Will be passed to pd.apply().
        
        Parameter
        ---------
        raw_title : str
            the raw title
        
        Returns
        -------
        clean_title : str
            the 'clean' title
            
        """
        clean_title = ''
        for char in raw_title:
            if char == ' ':
                clean_title += char
            elif char == '.':
                clean_title += ' '
            else:
                clean_title += char
            
        # Keep only alphanumeric chars, as well as spaces
        p = re.compile('[^a-zA-Z0-9_\s]')
        clean_title = p.sub('', clean_title)
            
        return clean_title.lower()
    
    df['clean_title'] = df['title'].apply(clean_title)
    return df

def ngrams(text, n=1):
    """
    Counts the number of times a contiguous sequence of n words appeared
    in the string specified.

    Parameters
    ----------
    n : int, optional (default 1)
        value of -grams

    Returns
    -------
    grams_dict : dictionary
        ngram string as keys and their counts as values
        
    """
    grams_list = []
    grams_dict = {}
    list_of_paragraphs = text.split('\n')
    for paragraph in list_of_paragraphs:
        words = paragraph.split()
        for index in range(0, len(words)):
            temp = words[index:index+n]
            if len(temp) == n:
                grams_list.append(' '.join(word for word in
                                           words[index:index+n]))
    for term in grams_list:
        if term not in grams_dict:
            grams_dict[term] = 1
        else:
            grams_dict[term] += 1

    sorted_grams = dict(sorted(grams_dict.items(),
                               key=lambda temp: (-temp[1], temp[0])))
    return sorted_grams

def plot_ngrams(text_, n=1, top_n=10):
    """
    Plots the ngrams using a horizontal bar graph.

    Parameters
    ----------
    n : int, optional (default 1)
        value of n-grams
    top_n : int, optional (default 10)
        number of terms to include in the graph

    Returns
    -------
    fig : matplotlib Figure
        horizontal bar graph
    """
    grams = ngrams(text_, n)
    count = 0
    data = []
    for key, value in grams.items():
        if count < top_n:
            data.append([value, key])
            count += 1
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    sort = sorted(data, key=lambda x: (-x[0], x[1]), reverse=True)
    y = [item[0] for item in sort]
    x = [item[1] for item in sort]
    df = pd.DataFrame({'words': y, 'counts': x})
    ax = df.plot.barh(x='counts', y='words', color='C0', ax=ax, legend=False)
    ax.set_title('Top {} Most Frequent Words'.format(top_n))
    ax.set_ylabel('WORDS')
    ax.set_xlabel('FREQUENCY')

def create_word_cloud(string):
    """
    Creates a word cloud from an input string using the WordCloud module.
    
    Parameter
    ---------
    string : str
        the string to create the word cloud from
        
    Return
    ------
    cloud : WordCloud object
    """
    mask = np.array(PILImage.open('figures/cloud1.png'))
    cloud = WordCloud(background_color="white", mask=mask, width=1000,
                     height=750,max_font_size=300,
                     min_font_size=5).generate(string)
    return cloud
    
def intra_to_inter(X, y, dist, r):
    """Compute intracluster to intercluster distance ratio
    
    Parameters
    ----------
    X : array
        Design matrix with each row corresponding to a point
    y : array
        Class label of each point
    dist : callable
        Distance between two points. It should accept two arrays, each 
        corresponding to the coordinates of each point
    r : integer
        Number of pairs to sample
        
    Returns
    -------
    ratio : float
        Intracluster to intercluster distance ratio
    """

    total_p = 0.0
    ctr_p = 0
    total_q = 0.0
    ctr_q = 0
    np.random.seed(11)
    
    for i, j in np.random.randint(low=0, high=len(y), size=[r,2]):
        if i == j:
            continue
        elif y[i] == y[j]:
            total_p += dist(X[i], X[j])
            ctr_p += 1
        else:
            total_q += dist(X[i], X[j])
            ctr_q += 1
            
    p = total_p / ctr_p
    q = total_q / ctr_q
    ratio = float(p / q)
    return ratio

def cluster_range(X, clusterer, k_start, k_stop, step=1, actual=None):
    """
    Accepts the design matrix, the clustering object, the initial and final
    values to step through, and, optionally, actual labels. Returns a 
    dictionary of the cluster labels, internal validation values and, 
    if actual labels is given, external validation values, for every k.
    
    Parameters
    ----------
    X : array
        the design matrix
    clusterer : clustering object
    k_start : int
        initial value
    k_stop : int
        final value
    step : int, optional
        step in values, default 1
    actual : array
        actual labels, default None
        
    Returns
    -------
    c_range : dict
    """
    
    amis, ars, ps, chs, iidrs, inertias, scs, ys = [], [], [], [], [], [], \
    [], []

    for k in range(k_start, k_stop+1, step):
        print("Clustering with k = {}...".format(k), end='')
        clusterer.n_clusters = k
        km = clusterer
        X_predict = km.fit_predict(X)
        chs.append(calinski_harabasz_score(X, X_predict))
        iidrs.append(intra_to_inter(X, X_predict, euclidean, 50))
        inertias.append(km.inertia_)
        scs.append(silhouette_score(X, X_predict))
        ys.append(X_predict)
        
        if type(actual) != type(None):
            amis.append(adjusted_mutual_info_score(actual, X_predict, 
                                                   average_method='max'))
            ars.append(adjusted_rand_score(actual, X_predict))
            ps.append(purity(actual, X_predict))
        print(' Done!')
        
    c_range = {}
    c_range['chs'] = chs
    c_range['iidrs'] = iidrs
    c_range['inertias'] = inertias
    c_range['scs'] = scs
    c_range['ys'] = ys
    
    # Optionally,
    if type(actual) != type(None):
        c_range['amis'] = amis
        c_range['ars'] = ars
        c_range['ps'] = ps
    return c_range

def plot_internal(inertias, chs, iidrs, scs):
    """
    Plot internal validation values.
    
    Parameters
    ----------
    inertias : list or array-like
    chs : list or array-like
    iidrs : list or array-like
    inertias : list or array-like
    """
    sns.set_context("talk")
    fig, ax = plt.subplots(figsize=(12, 8))
    ks = np.arange(2, len(inertias)+2)
    ax.plot(ks, inertias, '-o', label='SSE', ms=7)
    ax.plot(ks, chs, '-ro', label='CH', ms=7)
    ax.set_title('Internal Validation Criteria')
    ax.set_xlabel('$k$')
    ax.set_ylabel('SSE/CH')
    ax.set_xticks(ks)
    lines, labels = ax.get_legend_handles_labels()
    ax2 = ax.twinx()
    ax2.plot(ks, iidrs, '-go', label='Inter-intra', ms=7)
    ax2.plot(ks, scs, '-ko', label='Silhouette coefficient', ms=7)
    ax2.set_ylabel('Inter-Intra/Silhouette')
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines+lines2, labels+labels2)
    return ax

def plot_clusters(X, ys):
    """
    Plot clusters given the design matrix and cluster labels.
    
    Parameters
    ----------
    X : matrix
        the design matrix
    ys : list or array-like
        the cluster labels
    """
    k_max = len(ys) + 1
    k_one = k_max//4 + 2
    k_two = 2*k_max//4 + 2
    k_three = 3*k_max//4 + 2
    fig, ax = plt.subplots(4, k_max//4, dpi=150, sharex=True, sharey=True, 
                           figsize=(7, 4), subplot_kw=dict(aspect='equal'),
                           gridspec_kw=dict(wspace=0.01, hspace=0.5))
    for k,y in zip(range(2, k_max+1), ys):
        if k < k_one:
            ax[0][k-2].scatter(*zip(*X), c=y, s=1, alpha=0.8)
            ax[0][k-2].set_title('$k=%d$'%k)
        elif k < k_two:
            ax[1][k%k_one].scatter(*zip(*X), c=y, s=1, alpha=0.8)
            ax[1][k%k_one].set_title('$k=%d$'%k)
        elif k < k_three:
            ax[2][k%k_two].scatter(*zip(*X), c=y, s=1, alpha=0.8)
            ax[2][k%k_two].set_title('$k=%d$'%k)
        else:
            ax[3][k%k_three].scatter(*zip(*X), c=y, s=1, alpha=0.8)
            ax[3][k%k_three].set_title('$k=%d$'%k)
    return ax
    
def produce_kmeans_cluster(k):
    """
    Retrieve the kmeans cluster object, plot the output using TSNE, and
    assign the predicted targets to the DataFrame.
    
    Parameter
    ---------
    k : int
        number of clusters
        
    """
    y_predict_kmeans = {}
    sns.set_context('talk')
    y_predict_kmeans[k] = res_reddit['ys'][k-2]
    plt.scatter(X_reddit_new[:,0], X_reddit_new[:,1], c=y_predict_kmeans[k], 
                s=15)
    plt.title("K Means Clustering with k = {}".format(k))
    plt.show()
    
    for num_clusters in Counter(y_predict_kmeans[k]):
        print("Number of members in cluster {}: {}"\
              .format(num_clusters+1, 
                      Counter(y_predict_kmeans[k])[num_clusters]))

    clean_df['kmeans_{}'.format(k)] = y_predict_kmeans[k]

def illustrate_clusters(df, k):
    """
    Illustrate the clusters obtained using a word count bar graph, as well as
    a word cloud.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to take data and clusters from
    k : int
        number of clusters
        
    """
    for i in range(k):
        cluster_text = ' '.join(df[df['kmeans_{}'.format(k)]==i]\
                                ['clean_title']).split()
        cluster_text = ' '.join([word for word in cluster_text if \
                                 word not in new_stop_words])
        display(HTML('<h2 style="color: blue">Cluster {} of {}</h2>'\
                     .format(i+1, k)))
        display(HTML("<h4>Word Count of Cluster {}</h4>".format(i+1)))
        plot_ngrams(cluster_text, top_n=20)
        plt.show()
        display(HTML("<h4>Word Cloud of Cluster {}</h4>".format(i+1)))
        plt.figure(figsize=(15, 10))
        plt.imshow(create_word_cloud(cluster_text))
        plt.axis('off')
        plt.show()
        
def remove_til(str_):
    new_str = ' '.join([word for word in str_.split() if word != 'til'])
    return new_str
    
def produce_kmeans_cluster_til(k):
    y_predict_kmeans = {}
    sns.set_context('talk')
    y_predict_kmeans[k] = res_til['ys'][k-2]
    plt.scatter(X_til_new[:,0], X_til_new[:,1], c=y_predict_kmeans[k], s=15)
    plt.title("K Means Clustering of TIL posts with k = {}".format(k))
    plt.show()
    
    for num_clusters in Counter(y_predict_kmeans[k]):
        print("Number of members in cluster {}: {}"\
              .format(num_clusters+1, 
                      Counter(y_predict_kmeans[k])[num_clusters]))

    clean_df1.loc[:, 'kmeans_{}'.format(k)] = y_predict_kmeans[k]
    
def illustrate_clusters(df, k):
    for i in range(k):
        cluster_text = ' '.join(df[df['kmeans_{}'\
                                      .format(k)]==i]['no_til_title']).split()
        cluster_text = ' '.join([word for word in \
                                 cluster_text if word not in new_stop_words])
        display(HTML("<h2>Cluster {} of {}</h2>".format(i+1, k)))
        display(HTML("<h4>Word Count of Cluster {}</h4>".format(i+1)))
        plot_ngrams(cluster_text, top_n=20)
        plt.show()
        display(HTML("<h4>Word Cloud of Cluster {}</h4>".format(i+1)))
        plt.figure(figsize=(15, 10))
        plt.imshow(create_word_cloud(cluster_text))
        plt.axis('off')
        plt.show()