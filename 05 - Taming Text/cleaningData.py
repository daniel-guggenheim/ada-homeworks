import os
import re
import string

import nltk
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize, sent_tokenize

'''
This file contains the helper function getTokenMapFromData(). This function is executing
the same work than the cleaning that was done in Ex. 5.1. Therefore, it will not be
commented and explained, as you can find the explanation on part 5.1.

'''

def getTokenMapFromData(use_english_vocab):
    '''
    Import the data, clean it, and return a list of list of tokens.
    The list id corresponds to the clean data index. The list of tokens corresponds
    to the words in the email of the corresponding index.
    :param use_english_vocab: Boolean, true if we want to remove the words that are not in
    the english vocabulary list, false otherwise.
    :return: List of list of tokens per email
    '''
    print('Starting to import data')
    filename = os.path.join('hillary-clinton-emails', 'Emails.csv')
    df = pd.read_csv(filename)
    df = df.dropna(axis=0, how='any', subset=['ExtractedBodyText'])
    extractedBodyText = df['ExtractedBodyText']
    # extractedBodyText.fillna("", inplace=True)

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
    stopwords.append('Fwd')
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    punctuations = list(string.punctuation)
    punctuations.append("''")

    print('Creating tokens...')
    list_of_token_list = [get_tokens(mail, stopwords, english_vocab, punctuations, use_english_vocab) for mail in extractedBodyText5]
    mapping_index_to_tokens = {}
    df_index_list = list(df.index)
    print('list token len=',len(list_of_token_list))
    print('list index len=', len(df_index_list))
    for i in range(0,len(list_of_token_list)):
        mapping_index_to_tokens[df_index_list[i]] = list_of_token_list[i]
    print('Finished!')
    return mapping_index_to_tokens


def get_tokens(mail, stopwords, english_vocab, punctuations, use_english_vocab):
    '''
    Take an email text and transform it in a list of tokens.
    :param mail: mail that we want to transform in list of tokens
    :param stopwords:  previously initialized list of stopwords
    :param english_vocab: previously initialized list of english vocabulary
    :param punctuations: previously initialized list of punctuations
    :param use_english_vocab: Boolean, true if we want to filter the words that are not in
    the english vocabulary list, false otherwise
    :return: list of tokens
    '''
    tokenizer = RegexpTokenizer(r'\w+')
    tokenizer.tokenize('Eighty-seven miles to go, yet.  Onward!')
    tokens = [word for sent in sent_tokenize(mail) for word in word_tokenize(sent)]
    noStopWords = [word for word in tokens if word not in stopwords]

    #Code from https://stackoverflow.com/questions/23317458/how-to-remove-punctuation
    noStopWords = [i for i in noStopWords if i not in punctuations]
    noStopWords = [i.strip("".join(punctuations)) for i in noStopWords if i not in punctuations]
    if(use_english_vocab):
        englishWords = [word for word in noStopWords if word in english_vocab]
        return englishWords
    else:
        return noStopWords
