# Naive_Bayese
# # '''Naive Bayes classifier'''

Problem Statement:
You are hired as an AI expert in the development department of a telecommunications company.
The first thing on your orientation plan is a small project that your boss has assigned you for
the following given situation. Your supervisor has given away his private cell phone number on
too many websites and is now complaining about daily spam SMS. Therefore, it is your job to
write a spam detector in Python from scratch. In doing so, you are to implement a Naive Bayes
classifier yourself that can handle both bag-of-words (BoW) and tf-idf-features as input.
For the evaluation of your spam detector, an SMS collection is available as a dataset  this has
yet to be suitably split into train and test data. To keep the costs as low as possible and to
avoid problems with copyrights, your boss insists on a new development with Python. If you
need special Python modules, you can discuss their use with the instructor before submission.
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


Results
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
