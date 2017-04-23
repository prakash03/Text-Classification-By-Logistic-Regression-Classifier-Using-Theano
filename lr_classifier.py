# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 16:39:24 2016

@author: PRAKASH
"""
import numpy as np
import theano
import theano.tensor as T
#import codecs
from sentiment_reader import SentimentCorpus
#floatX = theano.config.floatX
import timeit

start_time=timeit.default_timer()

data = SentimentCorpus()    

n_classes = 2 #posiitive and negative classes
n_instances = data.train_y.shape[0]
n_feats = data.nr_features
n_epoches = 3000

# randomly generate training data
train_x = data.train_X
#train_y = data.train_y
train_y = np.reshape(data.train_y,data.train_y.shape[0])

# declare Theano symbolic variables
x = T.matrix("x")
y = T.ivector("y")
w = theano.shared(np.random.randn(n_feats,n_classes), name="w")
b = theano.shared(np.zeros(n_classes), name="b")

print("Initial model:")
print(w.get_value())
print(b.get_value())

# construct Theano expression graph
p_y_given_x = T.nnet.softmax(T.dot(x, w) + b)
xent = -T.mean(T.log(p_y_given_x)[T.arange(n_instances), y])
cost = xent + 0.01 * (w ** 2).sum()       # The cost to minimize
gw, gb = T.grad(cost, [w, b])             # Compute the gradient of the cost
y_pred = T.argmax(p_y_given_x, axis=1)
error = T.mean(T.neq(y_pred, y))

# compile
train = theano.function(inputs=[x,y],
          outputs=[error, cost],
          updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))

# train
for i in range(n_epoches):
    error, cost = train(train_x, train_y)
    #print 'Current error: %.4f | Current cost: %.4f' % (error, cost)

print("Final model:")
print(w.get_value())
print(b.get_value())

#Testing function
test   = theano.function(inputs=[x], outputs=[y_pred])
#Input to the test function          
testing = test(data.test_X)

testing = np.asarray(testing)

testing = np.reshape(testing,testing.shape[1])
#y test set
test_y_new= np.reshape(data.test_y,data.test_y.shape[0])

#predicting accuracy
accuracy=0
variable=1
for i in range(test_y_new.shape[0]):
    if(test_y_new[[i]]==testing[i]):
        accuracy+=1
        variable+=1
        if (variable == accuracy):
            break
        
print "Accuracy = ", ((float(accuracy)/testing.shape[0]))*100, "%"

#Claculating F_Score
tpos=0
tneg=0
fpos=0
fneg=0
for i in range(test_y_new.shape[0]):
    if(test_y_new[i]==0 and testing[i]==0):
        tpos+=1
    if(test_y_new[i]==1 and testing[i]==1):
        tneg+=1
    if(test_y_new[i]==0 and testing[i]==1):
        fneg+=1
    if(test_y_new[i]==1 and testing[i]==0):
        fpos+=1
        
prec    = (float(tpos)/(tpos+fpos))
rec     = (float(tpos)/(tpos+fneg))

f_score = (float(2*prec*rec)/(prec+rec))

print "Precision = ", prec
print "Recall =   ", rec
print "F score =  " , f_score

stop_time=timeit.default_timer()

print "Running time is ", ((stop_time-start_time)/60), "minutes"