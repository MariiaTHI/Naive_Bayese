# Naive_Bayese
# # '''Naive Bayes classifier'''
Problem Statement:
A customer of yours is dissatisfied with the quality of the speech recognition. After a conversation
with the customer, you found out that he dictates books. However, the speech recognition was not
built for this purpose. Accordingly, initial investigations on a book excerpt have also shown that
the language model used is not suitable. For building a better language model for the applicati-
on, you have asked your customer to provide a text from the book (see "Datasets and Resources")
In order to save costs and also to avoid problems with copyrights, your company has decided to
not use existing solutions for this project. A new program for estimating language models is now
to be programmed in Python from scratch. If you need special Python modules, you can discuss
their use with your instructor before submission

# Here are the general steps to implement a Naive Bayese Classifier from scratch on a dataset:
# 1. Load the dataset into a pandas DataFrame 
# 2. Convert the target variable to binary values (i.e., 0 for ham and 1 for spam).
# 3. Create a list of stop words.
# 4. Preprocess the data. Clean the input features by removing any punctuation, converting all text to lowercase, and removing any stop words (i.e., common words that are unlikely to help classify the message as spam or ham).
# 5. Split the preprocessed data into training and testing sets.
# 6. Build a model.(calculate tf for spam and ham, idf, tfidf for 2 classes, calculate propr probabilities)
# 7. Test Naive Bayese.
# 7. Evaluate using accuracy, precision and recall meausures



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
