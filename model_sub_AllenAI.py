# -*- sharath -*-
"""
Following program selects best option for a given set of multiple choice questions 
from 8th standard syllabus - Allen AI Kaggle Competition

#Just enter the path of the folder containing .txt files in "path" variable
# Path of the file containing uestions that need to be answered in "data" variable
# It takes 20 mins to run on 4GB mac
#Corpus is approx. 1500 wikipedia pages relevant to 8th standard syllabus - As downloaded from
Allen_get_wikipages.py'
"""
import numpy as np
import re
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from math import log
import pandas as pd

# prepare tfidf index for corpus (ck12 text books of relevant topics downloaded from web)
# read data
# get question from every row
# get the closest paragraphs for the question
# get the best matched answer for the closest paragraph

#read the data - Multiple choice questions for which answers to be generated
data = pd.read_csv("/Users/homw/Documents/petp/AllenAI/test_set.tsv", sep = "\t")
sub4 = pd.read_csv("/Users/homw/Documents/petp/AllenAI/sub4.csv")

#Corpus that is to be used
''' The following path has wiki pages relevant to 8th standard syllabus - As downloaded from
Allen_get_wikipages'''

# Select the corpus path
path = '/Users/homw/Documents/petp/AllenAI/wiki/'
#path = '/Users/homw/Documents/petp/AllenAI/ck12'



# preprocess the line. 
def preprocess(line):
    #remove stop words, all symbols and numbers and split the line in to words
    line = re.sub("[^a-zA-Z]"," ",line)
    line = line.lower().split()
    
    #remove stopwords
    stops = stopwords.words("english")
    line = [word for word in line if not word in stops]
    
    #stemming
    stemmer = PorterStemmer()
    line = [stemmer.stem(word) for word in line]
    
    return line
    
# Get closest paragraphs from the corpus given a qst and tfidf of corpus
def get_closest_para_for_qst(qst, para_tf, idf):
    matched_para = []
    for para_name, para in para_tf.items():
        w_in_para_score = 0
        for word in qst:
            if word in para:
                w_in_para_score += para_tf[para_name][word] * idf[word]
        
        if w_in_para_score > 0:
            matched_para.append((para, w_in_para_score))
    
    #Get best matched para for the qst. Paragraph with highest score are the best matched ones with the qst
    matched_para = sorted(matched_para, key = lambda k: k[1], reverse = True)
    return matched_para[:3] #Return top 3 matched paragraphs

# Read the corpus - ck12 text books
''' Compute TfIdf values for all the words in the corpus after preprocessing'''

vocab = set()
# Initialize a dictionary to keep {line1:{word1:tf, word2:tf...}...}

total_words = 0
para_tf = {}
num = 1
for fname in os.listdir(path):
    if fname.endswith(".txt"):
        #print(fname)
        file = os.path.join(path, fname)
        
        for index, line in enumerate(open(file)):
            '''if index == 6038:            
                print(line, index)
                
        with open(file, "rb") as f:
            for line in f:'''
                
            if len(line) > 20:
                line = preprocess(line)
    
                if len(line) > 5:
                    total_line_words = 0 #To keep the count of words in paragraph/line
                    dic = {}
                    for word in line:
                        vocab.add(word)
                        dic.setdefault(word,0) #add a word to dictionary only if it is not existing
                        dic[word] = dic[word]+1
                        total_words = total_words + 1
                        total_line_words += 1
    
                    # Compute term freq for each word in a paragraph
                    for word, count in dic.items():
                        dic[word] = 0.5 + 0.5*(count/max(dic.values()))
    
                    # store Tf values of each paragraph in a dictionary
                    para_name = "para"+ str(num)
                    para_tf[para_name] = dic
                    num += 1

# Compute idf values for all the words in vocabulary
idf = {}
for word in list(vocab):
    docs_has_word = 1
    for index,doc in para_tf.items():
        if word in doc:
            docs_has_word += 1
    idf[word] = log((len(para_tf)+1)/docs_has_word)
    

# Build predictions for a given set of multiple choices
prediction = []
missed = 0
for index, record in data.iterrows():
    #print(index, record)
    qst = preprocess(record["question"])
    closest_paras = get_closest_para_for_qst(qst, para_tf, idf) #Get only the paragraph, score is not required
    # Now check which of the options out of A,B,C,D scores highest with the best matched paragraphs of the Qst
    
    opt_A = preprocess(record["answerA"])
    score_A = 0
    for word in opt_A:
        for para, score in list(closest_paras):
            if word in para:
                score_A += para[word] * idf[word]
                
    
    opt_B = preprocess(record["answerB"])
    score_B = 0
    for word in opt_B:
        for para, score in list(closest_paras):
            if word in para:
                score_B += para[word] * idf[word]

                
    
    opt_C = preprocess(record["answerC"])
    score_C = 0
    for word in opt_C:
        for para, score in list(closest_paras):
            if word in para:
                score_C += para[word] * idf[word]

                
    
    opt_D = preprocess(record["answerD"])
    score_D = 0
    for word in opt_D:
        for para, score in list(closest_paras):
            if word in para:
                score_D += para[word] * idf[word]

    if all([score_A,score_B,score_C,score_D]) == 0:
        prediction.append("N")
        missed += 1
    else:
        prediction.append(["A","B","C","D"] [np.argmax([score_A,score_B,score_C,score_D])])
    if len(prediction)%500 == 0:
        print(len(prediction), missed) 


prediction1 = ["B" if p == "N" else p for p in prediction] #replace "N"
# Write prediction to submission file
pd.DataFrame({'id': list(data['id']), 'correctAnswer': prediction1})[['id', 'correctAnswer']].to_csv("sub5.csv", index = False)

# choosing 4 closest documents missed 14512 questions
len([1 for a,b in zip(sub4.correctAnswer, prediction1) if a == b])