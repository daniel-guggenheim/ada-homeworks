import pandas as pd
import numpy as np
import os
import re
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import PorterStemmer

## global variables
english_vocab = None
stopwords = None
re_word_filter = None

def getTokenMapFromData():
    print('Starting to import data')
    global english_vocab
    global stopwords
    global re_word_filter

    stopwords = nltk.corpus.stopwords.words('english')
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())

    # extend set of stopwords modals with modal verbs
    stopwords = stopwords + ['could', 'may', 'might', 'must', 'ought to', 'shall', 'would']

    emails_file = os.path.join('hillary-clinton-emails', 'Emails.csv')
    topic_modeling_df = pd.read_csv(emails_file)
    topic_modeling_df = topic_modeling_df.dropna(subset=['ExtractedBodyText'], how='all')

    print('Removing words too short...')
    re_word_filter = re_longer_than(4)

    print('Getting tokens...')
    stem_tokens = get_df_mapped_tokens(topic_modeling_df, 'RawText')

    print('Finished!')
    return stem_tokens


def re_longer_than(N):
    return re.compile('^[a-z]{' + '{0},'.format(N) + '}')


def preprocess_msg(msg):
    sentences = nltk.sent_tokenize(msg)
    del sentences[:6]
    del sentences[-7:]

    tokens = []
    for s in sentences:
        curr_tokens = nltk.word_tokenize(s)
        curr_tokens = [word for word in curr_tokens if word in english_vocab]
        curr_tokens = [word for word in curr_tokens if word not in stopwords]
        tokens = tokens + curr_tokens

    return tokens


def get_df_mapped_tokens(df, column_name):
    rowIdToEmailMap = {}

    def makeRowIdToRawTextMapping(row):
        rowIdToEmailMap[row.name] = row[column_name]

    df.apply(makeRowIdToRawTextMapping, axis=1)
    return {k: preprocess_msg(v) for k, v in rowIdToEmailMap.items()}