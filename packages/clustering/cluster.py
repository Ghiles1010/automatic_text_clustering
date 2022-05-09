import numpy as np
import pandas as pd
from wordcloud import WordCloud
# import from root folder
from ..utils.plot import plot_grid


def cluster_key_features(vectors, labels, feature_names, n=20):

    all_words = pd.Series(feature_names)
    sr = pd.Series(labels)
    groups = sr.groupby(sr)

    cluster_key_feature_dict = {}

    # vectors = vectors.todense()
    
    for key in groups.groups:
        index = groups.get_group(key).index
        group_vectors = vectors[index]
        cluster_key_feature = group_vectors.mean(axis=0)
        cluster_key_feature = np.asarray(cluster_key_feature).squeeze()
        cluster_key_feature = zip(all_words, cluster_key_feature)
        cluster_key_feature = sorted(cluster_key_feature, key=lambda kv: kv[1], reverse=True)
        cluster_key_feature = cluster_key_feature[:n]
        cluster_key_feature = dict(cluster_key_feature)
        
        cluster_key_feature_dict[key] = cluster_key_feature
    
    return cluster_key_feature_dict


def plot_cluster_key_features(vectors, labels, feature_names, max_feat=50):
    ckw = cluster_key_features(vectors, labels, feature_names)

    clouds = []
    for key in ckw:
        key_words = ckw[key]
        Cloud = WordCloud(background_color="white", max_words=max_feat).generate_from_frequencies(key_words)
        clouds.append(Cloud)

    plot_grid(clouds, figsize=(10,10), caption="Cluster")

    







        