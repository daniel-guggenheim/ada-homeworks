import pandas as pd
import numpy as np
import os
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
import string

## global variables
english_vocab = None
stopwords = None
re_word_filter = None

def getTokenMapFromData():
    print('Starting to import data')
    filename = os.path.join('hillary-clinton-emails', 'Emails.csv')
    df = pd.read_csv(filename)
    extractedBodyText = df['ExtractedBodyText']
    extractedBodyText.fillna("", inplace=True)

    print('Starting cleaning...')
    extractedBodyText2 = extractedBodyText.apply(lambda x: re.sub('<.*>', ' ', x))
    del extractedBodyText
    extractedBodyText3 = extractedBodyText2.apply(lambda x: re.sub('030311 .* \\n', '', x))
    del extractedBodyText2
    extractedBodyText4 = extractedBodyText3.apply(lambda x: re.sub('^STATE DEPT .* STATE-[0-9A-Z]+', ' ', x))
    extractedBodyText4 = extractedBodyText4.apply(lambda x: re.sub('Date: .* \\n', ' ', x))
    extractedBodyText4 = extractedBodyText4.apply(lambda x: re.sub('Doc No.: .* \\n', ' ', x))
    extractedBodyText4 = extractedBodyText4.apply(lambda x: re.sub('Case No.: .* \\n', ' ', x))
    extractedBodyText4 = extractedBodyText4.apply(lambda x: re.sub('Sent: .* \\n', ' ', x))
    extractedBodyText4 = extractedBodyText4.apply(lambda x: re.sub('To: .* \\n', ' ', x))
    extractedBodyText4 = extractedBodyText4.apply(lambda x: re.sub('^[A-Z] .* @ .* \\n', ' ', x))
    extractedBodyText4 = extractedBodyText4.apply(
        lambda x: re.sub('[A-Z]{1}[a-z]+, [A-Z]{1}[a-z]+ \d+, [0-9]+ \d+:\d+ [A-Z]{2}', ' ', x))
    extractedBodyText4 = extractedBodyText4.apply(lambda x: re.sub(' [a-zA-Z]{1} ', ' ', x))
    extractedBodyText4 = extractedBodyText4.apply(lambda x: re.sub(' [a-zA-Z]{2} ', ' ', x))
    del extractedBodyText3
    extractedBodyText5 = extractedBodyText4.apply(lambda x: re.sub('\n', ' ', x))
    del extractedBodyText4

    stopwords = nltk.corpus.stopwords.words('english')
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    punctuations = list(string.punctuation)
    punctuations.append("''")

    print('Creating tokens...')
    token_list = [get_tokens(mail, stopwords, english_vocab, punctuations) for mail in extractedBodyText5]
    print('Finished!')
    return token_list


def get_tokens(mail, stopwords, english_vocab, punctuations):
    tokenizer = RegexpTokenizer(r'\w+')
    tokenizer.tokenize('Eighty-seven miles to go, yet.  Onward!')
    tokens = [word for sent in sent_tokenize(mail) for word in word_tokenize(sent)]
    noStopWords = [word for word in tokens if word not in stopwords]
    #Code from https://stackoverflow.com/questions/23317458/how-to-remove-punctuation
    noStopWords = [i for i in noStopWords if i not in punctuations]
    noStopWords = [i.strip("".join(punctuations)) for i in noStopWords if i not in punctuations]
    # noStopWords = [token for token in noStopWords (lambda word: word not in ',.:;-', noStopWords)
    # englishWords = [word for word in noStopWords if word in english_vocab]
    return noStopWords




    # global english_vocab
    # global stopwords
    # global re_word_filter
    #
    # stopwords = nltk.corpus.stopwords.words('english')
    # english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    #
    # # extend set of stopwords modals with modal verbs
    # stopwords = stopwords + ['could', 'may', 'might', 'must', 'ought to', 'shall', 'would']
    #
    # emails_file = os.path.join('hillary-clinton-emails', 'Emails.csv')
    # topic_modeling_df = pd.read_csv(emails_file)
    # topic_modeling_df = topic_modeling_df.dropna(subset=['ExtractedBodyText'], how='all')
    #
    # print('Removing words too short...')
    # re_word_filter = re_longer_than(4)
    #
    # print('Getting tokens...')
    # stem_tokens = get_df_mapped_tokens(topic_modeling_df, 'RawText')
    #
    # print('Finished!')
    # return stem_tokens


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