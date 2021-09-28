import pandas as pd
import re
from nltk.corpus import stopwords
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('stopwords')


def import_merge():
    df = pd.read_csv("olist_orders_dataset.csv")
    df = df[df["order_status"] == "delivered"]
    df = df[["order_id", "customer_id"]]
    
    df_reviews = pd.read_csv("olist_order_reviews_dataset.csv")
    df_reviews = df_reviews[["order_id", "review_score", "review_comment_message"]]
    
    df = pd.merge(df, df_reviews, how="left")
    
    df = df.drop(["order_id", "customer_id"], axis=1)
    df = df.dropna()

    return df

def preprocess(df):
    df["review_comment_message"] = df["review_comment_message"].str.lower()
    df["review_comment_message"] = df["review_comment_message"].str.replace(r"[^\w\s]", "")
    df["review_comment_message"] = df["review_comment_message"].str.replace(r"\d+", "")

    return df

def remove_stopwords(df):
    stop_words = stopwords.words('portuguese')
    df["no_stop_words"] = df["review_comment_message"].apply(lambda x: " ".join([w for w in x.split() if w not in stop_words]))
    df = df.drop("review_comment_message", axis=1)
    return df