import nltk
from nltk.corpus import stopwords
# nltk.download('punkt')
import re
import string

def remove_newline(value):
    return ''.join(value.splitlines())

def remove_miss(value):
    cleanText = re.sub('[^A-Za-z0-9-,. ]+', '', value)
    return cleanText

def remove_whitespace(records):
  records_after_removal_whitespaces = " ".join(records.split())
  return  records_after_removal_whitespaces

def lowercasing(text):
  records = text.lower()
  return records

def textp(df):
    df = df.dropna()
    df_copy = df.copy()
    df_copy['clean_abstract'] = df_copy['Abstract'].apply(remove_newline)
    df_copy['clean_abstract'] = df_copy['clean_abstract'].apply(remove_miss)
    df_copy['clean_abstract'] = df_copy['clean_abstract'].apply(remove_whitespace)
    df_copy['clean_abstract'] = df_copy['clean_abstract'].apply(lowercasing)
    df_copy['Contributors'] = df_copy['Contributors'].apply(remove_newline)
    df_copy['Contributors'] = df_copy['Contributors'].apply(remove_miss)
    df_copy['Contributors'] = df_copy['Contributors'].apply(remove_whitespace)
    df_copy['Contributors'] = df_copy['Contributors'].apply(lowercasing)
    return df_copy
