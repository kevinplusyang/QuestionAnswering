import json
import numpy as np
import random
import re
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import nltk
import time
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import cross_validation
from __future__ import print_function
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import strin
from nltk import pos_tag, ne_chunk
import sys
reload(sys)
sys.setdefaultencoding('utf8') 

stopWords = set(stopwords.words('english'))
exclude = set(string.punctuation)

#read dataset for question classification training and testing
#data source:http://cogcomp.org/Data/QA/QC/
with open('train-data.tsv', 'r') as f:
    trainY = []
    trainX = []
    for line in f:
        trainY.append(line.split("\t")[3].replace("\n",""))
        trainX.append(word_tokenize(line.split("\t")[2]))

with open('test-data.tsv', 'r') as f:
    testY = []
    testX = []
    for line in f:
        testY.append(line.split("\t")[3].replace("\n",""))
        testX.append(word_tokenize(line.split("\t")[2]))
#function to extract feature of question  
#feture dictionary: first word, second word, first key word, second key word
def q_to_fdict(q):
    
    fdict = {}
#     question fisrt word: what, which, when, where, who, how, why, and unknown
    question_word = ['What','Which','When','Where','Who','How','Why']
    if q[0] in question_word:
        fdict['1st-word'] = q[0]
    else:
        fdict['1st-word'] = 'Unk'
    
    if get_keywords(q) != []:
        keyword = get_keywords(q)[0]  
    else:
        keyword = ''
    
    if len(get_keywords(q)) >1:
        keyword2 = get_keywords(q)[1]  
    else:
        keyword2 = ''

    try:
        fdict['2nd-word'] = q[1]
    except:
        print(q)
    fdict['key_word'] = keyword
    fdict['extra_word'] = keyword2
    return fdict    

#return keywords list of qeustion
def get_keywords(q):
    keywords = []
    #remove the first word in q (what/when/...) 
    q = q[1:]
    #remove stop words and punctuation
    for w in q:
        if (w not in stopWords) and ((w not in exclude) and (w != '``')):
            keywords.append(w)
    return keywords

#read SQuad dataset 
with open('training.json') as dataset_file:
    dataset_json = json.load(dataset_file)
    dataset_train = dataset_json['data']

    
with open('testing.json') as dataset_file:
    dataset_json = json.load(dataset_file)
    dataset_test = dataset_json['data'] 
  
with open('development.json') as dataset_file:
    dataset_json = json.load(dataset_file)
    dataset_dev = dataset_json['data'] 

#return question tokenize list
def get_question_list(dataset):
    q_list = []
    for article in dataset:
        for paragraph in article['paragraphs']:
            #get context of this paragraph
            context = paragraph['context']
            for qa in paragraph['qas']:
                #get question
                question  = word_tokenize(qa['question'])
                
                #remove useless question
                if len(question) <= 1:
                    continue
                    
                q_list.append(question)

    return q_list

#get all questions and predict question classification
q_list_train = get_question_list(dataset_train)
q_list_test = get_question_list(dataset_test)
q_list_dev = get_question_list(dataset_dev)

#get feature of question
S_trainX_fdict = [q_to_fdict(q) for q in q_list_train]
S_testX_fdict = [q_to_fdict(q) for q in q_list_test]
S_devX_fdict = [q_to_fdict(q) for q in q_list_dev]
S_whole_fdict = S_trainX_fdict + S_testX_fdict + S_devX_fdict
print(len(S_trainX_fdict), len(S_testX_fdict),len(S_devX_fdict),len(S_whole_fdict))

whole = whole_fdict + S_whole_fdict

#transfer text to numeric vector
Xdict = DictVectorizer()
whole_trans = Xdict.fit_transform(whole)
trainX_trans = whole_trans[:5452]
testX_trans = whole_trans[5452:5952]

cfier = LogisticRegression(solver='lbfgs', multi_class='multinomial')
cfier.fit(trainX_trans, trainY)
print (cfier.score(trainX_trans, trainY))

#Temporary solution: cross validation
res = cross_validation.cross_val_score(cfier, trainX_trans, trainY, cv=10)
print ("10 fold cross-validation accuracy:")
print (res)
print ("Average over folds")
print (sum(res) / float(len(res)))
print ("Accuracy on test data set")
print (cfier.score(testX_trans, testY))
cfier.coef_.tolist

index_train = len(whole_fdict) +len(S_trainX_fdict)
index_test = index_train + len(S_testX_fdict)

S_trainX_trans = whole_trans[5952:index_train]
S_testX_trans = whole_trans[index_train:index_test]
S_devX_trans = whole_trans[index_test:]

#get prediction for question classification in SQuad dataset
S_trainX_qc = cfier.predict(S_trainX_trans)
S_testX_qc = cfier.predict(S_testX_trans)
S_devX_qc = cfier.predict(S_devX_trans)

#return answer list and context list
def get_answer_context_list(dataset):
    a_list = []
    c_list = []
    for article in dataset:
        for paragraph in article['paragraphs']:
            #get context of this paragraph
            context = paragraph['context']
            
            for qa in paragraph['qas']:
                #get question
                question  = word_tokenize(qa['question'])
                
                #remove useless question
                if len(question) <= 1:
                    continue
                    
                #get answer
                answer  = qa['answers']
                a_list.append(answer)
                
                c_list.append(context)
   
    return a_list, c_list

#get answer and context list
a_list_train, c_list_train = get_answer_context_list(dataset_train)
a_list_test, c_list_test= get_answer_context_list(dataset_test)
a_list_dev, c_list_dev = get_answer_context_list(dataset_dev)

#return iob tagged list
def get_tagged(sentence):
    ne_tree = ne_chunk(pos_tag(word_tokenize(sentence)))
    iob_tagged = tree2conlltags(ne_tree)
    return iob_tagged

#label type: FACILITY, GPE, GSP, LOCATION, ORGANIZATION, PERSON
#return named entity chunk list
def get_chunk_ne(sentence, label):
    tree = ne_chunk(pos_tag(word_tokenize(sentence)))
    ne =[]
    for n in tree:
        if isinstance(n, nltk.tree.Tree):               
            if n.label() == label:
                ne.append(' '.join([child[0] for child in n]))

    return ne

#return NP chunk list
def get_chunk_NP(context):
    context = word_tokenize(context)
    sentence = pos_tag(context)
    #match pattarn to get NP
    grammar = "NP: {<DT>?<JJ>*<NN>}" 

    cp = nltk.RegexpParser(grammar) 
    tree = cp.parse(sentence)
    #extract NP
    np =[]
    for n in tree:
        if isinstance(n, nltk.tree.Tree):               
            if n.label() == 'NP':
                np.append(' '.join([child[0] for child in n]))

    return np

#return candidate answer list
def get_candidate_answer(c, q, qc):
    #get key words list of q
    w_in_q = word_tokenize(q)
    keywords_list =  get_keywords(w_in_q)
    #get context tokenized
    w_in_c_list = word_tokenize(c)
    #get index of overlap keywords
    index_keywords = []
    for i in range(len(w_in_c_list)):
        if w_in_c_list[i] in keywords_list:
            index_keywords.append(i)
    
    #slice window of context, window size : 7
    if len(index_keywords) > 0:
        min_index = index_keywords[0]
        max_index = index_keywords[-1]
        if min_index>6:
            if len(index_keywords)- max_index >6:
                w_in_c_slice = w_in_c_list[min_index-7:max_index+7]
            else:
                w_in_c_slice = w_in_c_list[min_index-7:]
        else:
            if len(index_keywords)- max_index >6:
                w_in_c_slice = w_in_c_list[:max_index+7]
            else:
                w_in_c_slice = w_in_c_list
    else:
        w_in_c_slice = w_in_c_list
        

    sentence = ' '.join(w_in_c_slice)
    #qc:['ABBR', 'DESC', 'ENTY', 'HUM', 'LOC', 'NUM']
    if len(w_in_c_slice) >0:
        if qc == 'NUM':
            answer_list = [str(s) for s in sentence.split() if s.isdigit()]

        #ne tag: FACILITY, GPE, GSP, LOCATION, ORGANIZATION, PERSON    
        elif qc == 'LOC':
            answer_list = get_chunk_ne(sentence, 'LOCATION') + get_chunk_ne(sentence, 'GPE') + get_chunk_ne(sentence, 'GSP')
        elif qc == 'HUM':
            answer_list = get_chunk_ne(sentence, 'PERSON') + get_chunk_ne(sentence, 'ORGANIZATION') 
           

        else:
            answer_list = get_chunk_NP(sentence) + get_chunk_ne(sentence, 'FACILITY')

    return answer_list

def getHighRankAnswer(answerList, question, context, windowSize):
    dic = {}
    contextDic = context.split()
    w_in_q = word_tokenize(question)
    keyword = get_keywords(w_in_q) 

    for answer in answerList:
        index = 0
        answerDic = answer.split()

        i = 0
        for i in range (len(contextDic) - len(answerDic) + 1):
            j = 0
            for j in range (len(answerDic)):
               
                if answerDic[j] != contextDic[j + i]:
                    break
              
            if j == (len(answerDic) - 1):
                if answerDic[len(answerDic) - 1] == contextDic[len(answerDic) - 1 + i]:
                 
                    index = i
                    break
        start = index - windowSize
        end = index + windowSize + len(answerDic)
        if start < 0:
            start = 0
        if end >= len(contextDic):
            end = len(contextDic) - 1
        score = 0
        i = start
        while i < end:
            if contextDic[i] in keyword:
                score = score + 1
            i = i + 1

        dic[answer] = score
    #remove key word answer in answer list
    for key in dic:
    for k in keyword:
        if k in key:
            dic[answer] = 0
            break
                
    currentMax = 0
    currentAnswer = ""
    for key in dic:
        if dic[key] > currentMax:
            currentMax = dic[key]
            currentAnswer = key
    if currentMax == 0:
        currentAnswer = answerList[0]
        
    return currentAnswer

#predict answer
def predict(c, q, qc):

    answer_list = get_candidate_answer(c, q, qc)
    
    m = len(answer_list) 
    
    answer = ''
    if m == 0:
        answer = ''
        
    elif m ==1:
        answer = answer_list[0]
    else:
        answer = getHighRankAnswer(answer_list, q, c, 5)
        print(answer)
           
    return answer

with open('development.json') as dataset_file:
    dataset_json = json.load(dataset_file)
    dataset = dataset_json['data']

#generate prediction
prediction = {}
i = 0
for article in dataset:
    for paragraph in article['paragraphs']:
    	#get context of this paragraph
        context = paragraph['context']

        for qa in paragraph['qas']:
        	#get question
            question  = qa['question']
            #get id of question
            q_id = qa['id']
            #get answer type
            qc = S_devX_qc[i]
            answer = predict(context, question, qc)
            i = i +1
            #add to prediction
            prediction[q_id] = answer

#output prediction
with open('prediction_dev.json', 'w') as outfile:
    json.dump(prediction, outfile)

with open('testing.json') as dataset_file:
    dataset_json = json.load(dataset_file)
    dataset = dataset_json['data']

#generate prediction
prediction = {}
i = 0
for article in dataset:
    for paragraph in article['paragraphs']:
    	#get context of this paragraph
        context = paragraph['context']

        for qa in paragraph['qas']:
        	#get question
            question  = qa['question']
            #get id of question
            q_id = qa['id']
            #get answer type
            qc = S_testX_qc[i]
            answer = predict(context, question, qc)
            i = i +1
            #add to prediction
            prediction[q_id] = answer

#output prediction
with open('prediction_test2.json', 'w') as outfile:
    json.dump(prediction, outfile)
                
