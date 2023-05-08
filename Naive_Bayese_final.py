#!/usr/bin/env python
# coding: utf-8

# # Progress of implementation  report

# # '''Naive Bayes classifier'''

# Here are the general steps to implement a Naive Bayese Classifier on a dataset:
# 1. Load the dataset into a pandas DataFrame 
# 2. Convert the target variable to binary values (i.e., 0 for ham and 1 for spam).
# 3. Create a list of stop words.
# 4. Preprocess the data. Clean the input features by removing any punctuation, converting all text to lowercase, and removing any stop words (i.e., common words that are unlikely to help classify the message as spam or ham).
# 5. Split the preprocessed data into training and testing sets.
# 6. Build a model.(calculate tf for spam and ham, idf, tfidf for 2 classes, calculate propr probabilities)
# 7. Test Naive Bayese.
# 7. Evaluate using accuracy, precision and recall meausures.


import pandas as pd
import string
import re
import math


data_path = "/smsspamcollection/SMSSpamCollection"

# Load the dataset into a pandas DataFrame.
df = pd.read_csv(data_path, sep='\t', header=None, names=['label', 'message'])

# Convert labels to binary values
df['label'] = df['label'].apply(lambda x: 1 if x == 'spam' else 0)


# # 2. Creating a list of stop words to remove. Create a vocabulary of all unique words in the spam and non-spam dataset

# ''' This can be done by sorting the vocabulary by frequency in the training set, and defining the top 10–100 vocabulary entries
# as stop words, or alternatively by using one of the many predefined stop word lists
# available online.  Then each instance of these stop words is simply removed from
# both training and test documents as if it had never occurred'''


#creating a vocabulary of all unique words in df
words_df = []
for i in range(len(df)):
    message = df['message'].iloc[i]
    #delete any leading or trailing whitespace in a message
    words = message.strip().split()
    for word in words:
        words_df.append(word.lower())


#Frequencies of unique words
def get_vocab_freq(words):
    vocab_freq = {}
    for word in words:
        if word in vocab_freq:
            vocab_freq[word] += 1
        else:
            vocab_freq[word] = 1
    return vocab_freq

vocab_freq_df = get_vocab_freq(words_df)
sorted_vocab_df = dict(sorted(vocab_freq_df.items(), key=lambda item: item[1], reverse=True))

#list of stop words
stop_words = []
for k,v in sorted_vocab_df.items():
    if v > 382:
        stop_words.append(k)
print(stop_words)


# # 3. Preprocess data and Stop words removal.
'''Dataset need to be lowercased, cleaned, stop words should be removed'''
stop_words = ['to', 'i', 'you', 'a', 'the', 'u', 'and', 'is', 'in', 'my', 'for', 'your', 'of', 'me', 'have', 'call', 'on', 'are', 'that', 'it', '2', 'so', 'but', 'or', 'not', 'at', 'can', 'ur']
def preprocess(text):
    # remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    #delete any leading or trailing whitespace in a message
    words = text.strip().split()
    #keeping all uppercase words and lower Capital letters
    word_list = []
    for word in words:
        if word.isupper() and len(word) > 1:
            word_list.append(word)
        else:
            word_list.append(word.lower())

    #stop word removal
    text = ' '.join([word for word in word_list if word not in stop_words])
    return text


df['message'] = df['message'].apply(preprocess)


# # 4. Split data into test and training set

# randomly shuffle the rows of the dataframe
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# calculate the size of the train and test sets
train_size = int(0.8 * len(df))
test_size = len(df) - train_size

# split the dataframe into train and test sets
train_df = df[:train_size]
test_df = df[train_size:]

# print the size of the train and test sets
print('Train set size:', len(train_df))
print('Test set size:', len(test_df))


#checking if we have both classes in train and test set
print(train_df['label'].unique(), test_df['label'].unique())
print(len(train_df[train_df['label']==1]), len(train_df[train_df['label']==0]))
print(len(test_df[test_df['label']==1]), len(test_df[test_df['label']==0]))


# # Creating 2 dictionaries of unique words for spam and ham

#Create a list of words for spam messages in train set
words_train_spam = []
for i in range(len(train_df[train_df['label']==1])):
    message_spam = train_df[train_df['label']==1]['message'].iloc[i]
    #delete any leading or trailing whitespace in a message
    words_spam = message_spam.strip().split()
    for word in words_spam:
        words_train_spam.append(word)


#Create a list of words for non-spam messages in train set
words_train_ham = []
for i in range(len(train_df[train_df['label']==0])):
    message_ham = train_df[train_df['label']==0]['message'].iloc[i]
    #delete any leading or trailing whitespace in a message
    words_ham = message_ham.strip().split()
    for word in words_ham:
        words_train_ham.append(word)
        
print(f"Number of words in train set for spam {len(words_train_spam)} and non-spam {len(words_train_ham)}")


# # Tf  Term frequencies

# tf ( t, d ) = n / N  #count of t in d / number of words in d
#     
# where tf is the term frequency function
#       t is the term/ word
#       d is the document
#       n is the number of occurences of t in d
#       N is the number of occurences of t in all documents
# 


#Frequencies of unique words in set
def get_vocab_freq(words):
    vocab_freq = {}
    for word in words:
        if word in vocab_freq:
            vocab_freq[word] += 1
        else:
            vocab_freq[word] = 1
    return vocab_freq


# In[91]:


#Frequencies of unique words for spam and non-spam
vocab_freq_train_spam = get_vocab_freq(words_train_spam)
vocab_freq_train_ham = get_vocab_freq(words_train_ham)
print(f"The number of unique words in training set for spam {len(vocab_freq_train_spam)} and non-spam {len(vocab_freq_train_ham)}")


#Calculating |V| - the total number of unique words in the training set for both classes
#Build the set of all unique words
set_spam_train = set(words_train_spam)
set_ham_train = set(words_train_ham)
unique_words = set_spam_train | set_ham_train
V = len(unique_words)
print(f"The length of unique words in train vocabulary is {V}")



#calculating prior probability for spam and non spam classes 
prior_prob_spam = len(train_df[train_df['label']==1])/len(train_df) 
prior_prob_ham = len(train_df[train_df['label']==0])/len(train_df)
print(f"Prior probabilities for spam {prior_prob_spam} and non-spam {prior_prob_ham} messages")


# # Inverse Document Frequency. IDF.
'''idf ( t, d ) = log ( D / { d ∈ D : t ∈ d } + 1)
when we have a large corpus size say N=10000, the IDF value explodes. 
So to dampen the effect we take the log of IDF.
where idf is the inverse document frequency function t is the term/ word d is the document D 
is the total number of documents { d ∈ D : t ∈ d } denotes the number of documents in which t occur'''

def idf(term, documents):
    N = len(documents)
    df = sum(1 for doc in documents if term in doc)
    return math.log(N / df)


# # Creating tf-idf dataframe

laplace = 1
table = {'words': [], 'tf_spam': [], 'tf_ham': [], 'loglikelihood': [], 'idf': [], 'tf_idf_spam':[], 'tf_idf_ham':[]}
for word in unique_words:
    probability_word_in_spam = (vocab_freq_train_spam.get(word,0)+laplace)/(len(words_train_spam)+V)
    probability_word_in_ham = (vocab_freq_train_ham.get(word,0)+laplace)/(len(words_train_ham)+V)
    loglikelihood = math.log(probability_word_in_spam/probability_word_in_ham)
    
    #calculating idf
    idf_score = idf(word, train_df['message'])
    
    # add values to the dictionaries
    table['words'].append(word)
    table['tf_spam'].append(probability_word_in_spam)
    table['tf_ham'].append(probability_word_in_ham)
    table['loglikelihood'].append(loglikelihood)
    table['idf'].append(idf_score)
    table['tf_idf_spam'].append(idf_score*probability_word_in_spam)
    table['tf_idf_ham'].append(idf_score*probability_word_in_ham)
else:
    loglikelihood = 0
    

# create a DataFrame from the dictionary
tf_idf = pd.DataFrame(table)
tf_idf.iloc[0:50]


#creating a list to take a look at the messages that were wrong predicted 
wrong_predicted_FP = []
wrong_predicted_FN = []
#counts
count_TP = 0
count_TN = 0
count_FP = 0
count_FN = 0
#define prior probabilities
log_prior = math.log(prior_prob_spam/prior_prob_ham)

for i in range(0, len(test_df)):
    #real label
    y = test_df['label'].iloc[i]
    #get the message
    message = test_df['message'].iloc[i] 
    #split the message into words
    words = message.strip().split()
    pred_score = log_prior
    y_pred = None
    #loop through words in that messages    
    for word in words:
        if word in unique_words:
            #loglikelihood = tf_idf[tf_idf['words']==word]['loglikelihood'].item()
            loglikelihood = math.log(tf_idf[tf_idf['words']==word]['tf_idf_spam'].item()/tf_idf[tf_idf['words']==word]['tf_idf_ham'].item())
        else:
            loglikelihood = 0
               
        pred_score += loglikelihood
    
        
    if pred_score > 0:
        y_pred = 1
    else:
        y_pred = 0
        
        
    if y_pred == 1:
        if y_pred == y:
            count_TP += 1
        else:
            count_FP += 1
            wrong_predicted_FP.append((y, y_pred, message))
    else:
        if y_pred == y:
            count_TN += 1
        else:
            count_FN += 1
            wrong_predicted_FN.append((y,y_pred, message))
            
print(f"TP: {count_TP}, TN: {count_TN}, FP: {count_FP}, FN: {count_FN}")



precision = count_TP/(count_TP+count_FP)
recall = count_TP/(count_TP+count_FN)
accuracy = (count_TP+count_TN)/(count_TP+count_TN+count_FP+count_FN)
print(f"Accuracy={accuracy}, precision={precision}, recall={recall}.")


# Stop words removing does not play a huge role, keeping uppercase words makes a slight improvement in accuracy.
# 
# without stop word removal 
# Accuracy=0.97847533632287, precision=0.9642857142857143, recall=0.8766233766233766.
# TP: 135, TN: 956, FP: 5, FN: 19
# 
# with stop word removal and lowercase all words
# Accuracy=0.97847533632287, precision=0.9642857142857143, recall=0.8766233766233766.
# TP: 135, TN: 956, FP: 5, FN: 19
# 
#                 
# with removing stop words and keeping UPPER case, Precision and accuracy are a bit higher 
# Accuracy=0.9802690582959641, precision=0.9782608695652174, recall=0.8766233766233766.
# TP: 135, TN: 958, FP: 3, FN: 19
