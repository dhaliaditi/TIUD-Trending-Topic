#!/usr/bin/python2
# -*-coding:utf-8-*-

import lda
import numpy as np
import re
import StopWords
import scipy.stats

stop_word_list = StopWords.stop_word_list


def text_parse(big_string):
  
    list_of_tokens = re.split(r'\W*', big_string)
    return [tok.lower() for tok in list_of_tokens if len(tok) > 2]


def create_vocab_list():

    vocab_list = []
    with open('dict.txt') as dict:
        vocab_list = [word.lower().strip() for word in dict if (word.lower().strip() + ' ' not in stop_word_list)]
    return vocab_list


def normalize(mat):
   
    row_normalized_mat = []
    for row_mat in mat:
        normalized_row = []
        row = np.array(row_mat).reshape(-1, ).tolist()
        row_sum = sum(row)
        for item in row:
            if row_sum != 0:
                normalized_row.append(float(item) / float(row_sum))
            else:
                normalized_row.append(0)
        row_normalized_mat.append(normalized_row)
    return row_normalized_mat


def get_sim(t, i, j, row_normalized_dt):
   
    sim = 1.0 - abs(row_normalized_dt[i][t] - row_normalized_dt[j][t])
   
    return sim


def get_Pt(t, samples, tweets_list, friends_tweets_list, row_normalized_dt, relationship):
  
    Pt = []
    for i in xrange(samples):
        friends_tweets = friends_tweets_list[i]
        temp = []
        for j in xrange(samples):
            if relationship[j][i] == 1:
                if friends_tweets != 0:
                    temp.append(float(tweets_list[j]) / float(friends_tweets) * get_sim(t, i, j, row_normalized_dt))
                else:
                    temp.append(0.0)
            else:
                temp.append(0.0)
        Pt.append(temp)
    return Pt


def get_TRt(gamma, Pt, Et, iter=1000, tolerance=1e-16):
    
    TRt = np.mat(Et).transpose()
    old_TRt = TRt
    i = 0
    # np.linalg.norm(old_TRt,new_TRt)
    while i < iter:
        TRt = gamma * (np.dot(np.mat(Pt), TRt)) + (1 - gamma) * np.mat(Et).transpose()
        euclidean_dis = np.linalg.norm(TRt - old_TRt)
        # print 'dis', dis
        if euclidean_dis < tolerance:
            break
        old_TRt = TRt
        i += 1
    return TRt


def get_doc_list(samples):
   
    doc_list = []
    for i in xrange(1, samples + 1):
        with open('tweet_cont/tweet_cont_%d.txt' % i) as fr:
            temp = text_parse(fr.read())
        word_list = [word.lower() for word in temp if (word + ' ' not in stop_word_list and not word.isspace())]
        doc_list.append(word_list)
    return doc_list


def get_feature_matrix(doc_list, vocab_list):
   
    feature_matrix = []
   
    word_index = {}
    for i in xrange(len(vocab_list)):
        word_index[vocab_list[i]] = i
    for doc in doc_list:
        temp = [0 for i in xrange(len(vocab_list))]
        for word in doc:
            if word in word_index:
                temp[word_index[word]] += 1
        feature_matrix.append(temp)
    return feature_matrix


def get_tweets_list():
    
    tweets_list = []
    with open('number_of_tweets.txt') as fr:
        for line in fr.readlines():
            tweets_list.append(int(line))
    return tweets_list


def get_relationship(samples):
   
    relationship = []
    for i in xrange(1, samples + 1):
        with open('follower/follower_%d.txt' % i) as fr:
            temp = []
            for line in fr.readlines():
                temp.append(int(line))
        relationship.append(temp)
    return relationship


def get_friends_tweets_list(samples, relationship, tweets_list):
    
    friends_tweets_list = [0 for i in xrange(samples)]
    for j in xrange(samples):
        for i in xrange(samples):
            if relationship[i][j] == 1:
                friends_tweets_list[j] += tweets_list[i]
    return friends_tweets_list

    
def get_We(Pt,CT,max_iter=100,gamma=0.5,beta=0.5,tolerance=1e-16):
    for i in xrange(Pt):
        for j in xrange(Pt):
            PT_sum[j] += gamma*PT[i][j]
    PT_sum.sort()
    for i in xrange(CT):
        for j in xrange(CT):
            CT_sum[j]+=beta*CT[i][j]
    CT_sum.sort()
    We_sum =PT_sum+CT_sum    
    return We_sum

def get_user_list():
    
    user = []
    with open('user_id.txt') as fr:
        for line in fr.readlines():
            user.append(line)
    return user


def  get_CT(follower,friends_tweets_list,gamma=0.2, tolerance=1e-16):
    CT=[]
    or i in xrange(1, samples + 1):
        with open('follower(i)/follower(j)_%d.txt' % i) as fr:
            temp = []
            for line in fr.readlines():
                temp.append(int(line))
        CT.append(temp)

def get_TR(topics, samples, tweets_list, friends_tweets_list, row_normalized_dt, col_normalized_dt, relationship,
           gamma=0.2, tolerance=1e-16):
   
    TR = []
    for i in xrange(topics):
        Pt = get_Pt(i, samples, tweets_list, friends_tweets_list, row_normalized_dt, relationship)
        Et = col_normalized_dt[i]
        TR.append(np.array(get_TRt(gamma, Pt, Et, tolerance)).reshape(-1, ).tolist())
    return TR




def print_topics(model, vocab_list, n_top_words=5):
  
    topic_word = model.topic_word_
    # print 'topic_word',topic_word
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab_list)[np.argsort(topic_dist)][:-n_top_words:-1]
        print('Topic {}: {}'.format(i + 1, ' '.join(topic_words)))


def get_TR_using_DT(We_sum, samples, topics=5, gamma=0.2, tolerance=1e-16):
    
    row_normalized_we = normalize(We_sum)
   
    col_normalized_we_array = np.array(normalize(We_sum.transpose()))
    col_normalized_we = col_normalized_we_array.reshape(col_normalized_we_array.shape).tolist()
    tweets_list = get_tweets_list()
    relationship = get_relationship(samples)
    friends_tweets_list = get_friends_tweets_list(samples, relationship, tweets_list)
    user = get_user_list()
    TR = get_TR(topics, samples, tweets_list, friends_tweets_list,col_normalized_we, col_normalized_we_array, relationship,
                gamma, tolerance)
    for i in xrange(topics):
        print TR[i]
        print user[TR[i].index(max(TR[i]))]
    TR_sum = get_TR_sum(TR, samples, topics)
    return TR, TR_sum





def get_doc_topic_distribution_using_lda_model(model, feature_matrix):
    
    return model.transform(np.array(feature_matrix), max_iter=100, tol=0)


def using_lda_model_test_other_data(topics=5, n_iter=100, num_of_train_data=10, num_of_test_data=5, gamma=0.2,
                                    tolerance=1e-16):
   
    model, vocab_list = get_lda_model(samples=num_of_train_data, topics=topics, n_iter=n_iter)
    dt = model.doc_topic_
    print_topics(model, vocab_list, n_top_words=5)
    TR, TR_sum = get_TR_using_DT(dt, samples=num_of_train_data, topics=topics, gamma=gamma, tolerance=tolerance)
    doc_list = get_doc_list(samples=num_of_test_data)
    feature_matrix = get_feature_matrix(doc_list, vocab_list)
    dt = get_doc_topic_distribution_using_lda_model(model, feature_matrix)
 
    doc_user = np.dot(dt, TR)
    user = get_user_list()
    for i, doc in enumerate(doc_user):
        print user[i], user[list(doc).index(max(doc))]


def xtwitter_rank(topics=5, n_iter=100, samples=30, gamma=0.2, tolerance=1e-16):
    
    model, vocab_list = get_lda_model(samples, topics, n_iter)
    
    print_topics(model, vocab_list, n_top_words=5)
    
    dt = np.mat(model.doc_topic_)
    TR, TR_sum = get_TR_using_DT(dt, samples, topics, gamma, tolerance)


def main():
    xtwitter_rank()
   

if __name__ == '__main__':
    main()
