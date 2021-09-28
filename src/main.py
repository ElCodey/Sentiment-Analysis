import pandas as pd
import re
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


nltk.download('stopwords')

def undersample_split_vectorise(df):
    df_5 = df[df["review_score"] == 5].head(2717)
    df_4 = df[df["review_score"] == 4].head(2717)
    df_3 = df[df["review_score"] == 3].head(2717)
    df_2 = df[df["review_score"] == 2].head(2717)
    df_1 = df[df["review_score"] == 1].head(2717)
    df = pd.concat([df_5, df_4, df_3, df_2, df_1], ignore_index=True)
    
    vectorizer = CountVectorizer(max_features=3000)
    X = df["no_stop_words"]
    y = df["review_score"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, stratify=y, random_state=1)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.fit_transform(X_test)
    
    return  X_train, X_test, y_train, y_test

def oversample(df):
    sampler = RandomOverSampler(random_state=1)
    vectorizer = CountVectorizer(max_features=3000)
    X = df["no_stop_words"]
    y = df["review_score"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, stratify=y, random_state=1)
    X_train, y_train = sampler.fit_resample(X_train, y_train)
    X_train = vectorizer.fit_transform(X_train).toarray()
    X_test = vectorizer.fit_transform(X_test).toarray()
    
    
    return  X_train, X_test, y_train, y_test

def split(df):
    X = df["no_stop_words"]
    y = df["review_score"]
    vectorizer = CountVectorizer(max_features=3000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1)

    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.fit_transform(X_test)
    return X_train, X_test, y_train, y_test

def forest_model(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc_score = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    
    return acc_score, class_report, matrix

def log_model(X_train, X_test, y_train, y_test):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc_score = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    
    return acc_score, class_report, matrix


def grad_model(X_train, X_test, y_train, y_test):
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc_score = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    
    return acc_score, class_report, matrix

def oversample_2(df):
    df_sample = df[["no_stop_words", "review_score"]]
    train_df, test_df = train_test_split(df_sample, train_size=0.80)
    vectorizer = CountVectorizer(max_features=3000)
    X_sample = vectorizer.fit_transform(train_df["no_stop_words"])
    y_sample = train_df["review_score"]
    sampler = RandomOverSampler(random_state=1)
    X_train_os, y_train_os = sampler.fit_resample(X_sample, y_sample)
    X_test = vectorizer.fit_transform(test_df["no_stop_words"])
    y_test = test_df["review_score"]

    return X_train_os, X_test, y_train_os, y_test
