from preprocess import preprocess, load_spacy
from packages.clustering.Kmeans import Kmeans
from packages.clustering.vectorize import lsa
from packages.clustering.cluster import plot_cluster_key_features
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

nlp = load_spacy()


def read_data():
    df = pd.read_json("dialogsum/train.jsonl", lines=True)
    df.drop(["fname"], axis=1,inplace=True)
    df['topic'] = df.topic.apply(lambda x : x.lower())
    return df


def select_topics(df, nb_topics=10):
    # top_topics = sorted_topics.index[:nb_topics]
    
    # df.topic = df.topic.apply(lambda x : "food" if "food" in x else x)
    # df.topic = df.topic.apply(lambda x : "shopping" if "shopping" in x else x)
    # df.topic = df.topic.apply(lambda x : "job" if "job" in x else x)

    top_topics = ["interview", "restaurant"]
    df = df[df.topic.isin(top_topics)].reset_index(drop=True)
    return df




def main():

    df = read_data()
    df = select_topics(df)

    print("Preprocessing data...")
    df['summary'] = preprocess(df.summary, person_re="#.+?#", nlp=nlp)

    print("Vectorizing data...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    tf_idf_vectors = vectorizer.fit_transform(df.summary)
    feature_names = vectorizer.get_feature_names()

    print("Performing LSA...")
    lsa_vectors = lsa(tf_idf_vectors)

    print("Clustering data...")
    kmeans = Kmeans()
    kmeans.fit(lsa_vectors)

    plot_cluster_key_features(tf_idf_vectors, kmeans.km.labels_, feature_names)
    




if __name__ == '__main__':
    main()