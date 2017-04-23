# Text-Classification-By-Logistic-Regression-Classifier-Using-Theano
Claasification of Text using Logistics Regression

A Machine Learning concept.

Here, we have used two dataset files namely positive.review and negative.review. Both positive review and negative review act as respective classes, hence, the categories.

Each file (positive.review and negative.review) contains words followed by the frequency. Each line in each file represents a reviewed document containing related words and frequencies. Each line gets over with either #label:#positive or #label:negative.

The positive.review and negative.review files contain word frequencies in each line related to positive and negative reviewed document.

For example,

good:2 (a word good containing the frequency 2 in a line that represents to a document content)

The Logistic Regression Classifier divides the files into two two datasets namely training and testing set. The training set "trains" the algorithm to "test" the testing set in terms of matching the prediction with the actual classification values. The classification is based on two classes with values either 1 (positive) or 0 (negative).

This probabilistic approach of training and testing is done using Theano toolkit that does the softmax() and argmax().

The following are the implementation files:

sentiment_reader.py : This file takes the files as input, parses and makes the train and test datasets.

lr_classifier.py : This file takes datasets as input and calculates the Precision, Recall, Accuracy and F1 score.

