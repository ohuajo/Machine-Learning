
# coding: utf-8

# In[58]:


import numpy
import scipy.io
import math
from math import*
import geneNewData
import pandas

def main():
    myID='1417'
    geneNewData.geneData(myID)
    Numpyfile0 = scipy.io.loadmat('digit0_stu_train'+myID+'.mat')
    Numpyfile1 = scipy.io.loadmat('digit1_stu_train'+myID+'.mat')
    Numpyfile2 = scipy.io.loadmat('digit0_testset'+'.mat')
    Numpyfile3 = scipy.io.loadmat('digit1_testset'+'.mat')
    train0 = Numpyfile0.get('target_img')
    train1 = Numpyfile1.get('target_img')
    test0 = Numpyfile2.get('target_img')
    test1 = Numpyfile3.get('target_img')
    print([len(train0),len(train1),len(test0),len(test1)])
    print('Your trainset and testset are generated successfully!')
    return(train0, train1, test0, test1)


train0, train1, test0, test1 = main()

#Checking the dimension of the data
print(train0.shape, train1.shape, test0.shape, test1.shape)


# TASK 1:

# In[59]:


#Here I generated a 2-d data from the 3-d data by multiplying the rows and columns of the pixels each image. 
# This resulted in 784 column datsets
train0_2d = numpy.transpose(train0.reshape((train0.shape[1]*train0.shape[2]), train0.shape[0]))
train1_2d = numpy.transpose(train1.reshape((train1.shape[1]*train1.shape[2]), train1.shape[0]))
test0_2d = numpy.transpose(test0.reshape((test0.shape[1]*test0.shape[2]), test0.shape[0]))
test1_2d = numpy.transpose(test1.reshape((test1.shape[1]*test1.shape[2]), test1.shape[0]))

# Checking that the right 2-d arrays of 784 columns are obtained.
print(train0_2d.shape, train1_2d.shape, test0_2d.shape, test1_2d.shape)


# In[60]:


#Here the mean and standard deviations are extracted from the 784 column arrays to produce two column arrays
train0_features = numpy.zeros((5000, 2))
train0_features[:,0] = numpy.mean(train0_2d, axis=1)
train0_features[:,1] = numpy.std(train0_2d, axis=1)

train1_features = numpy.zeros((5000, 2))
train1_features[:,0] = numpy.mean(train1_2d, axis=1)
train1_features[:,1] = numpy.std(train1_2d, axis=1)

test0_features = numpy.zeros((980, 2))
test0_features[:,0] = numpy.mean(test0_2d, axis=1)
test0_features[:,1] = numpy.std(test0_2d, axis=1)

test1_features = numpy.zeros((1135, 2))
test1_features[:,0] = numpy.mean(test1_2d, axis=1)
test1_features[:,1] = numpy.std(test1_2d, axis=1)

# Checking that the right 2-d arrays are obtained - i.e. two column arrays.
print(train0_features.shape, train1_features.shape, test0_features.shape, test1_features.shape)


# TASK 2

# In[61]:


#Training Data
# Generate Mean of feature1(mean) and feature2 (standard deviation) for digit0 for the training dataset
mean_mean0 = numpy.mean(train0_features[:,0])
mean_std0 = numpy.mean(train0_features[:,1])
# Generate Mean of feature1(mean) and feature2 (standard deviation) for digit1 for the training dataset
mean_mean1 = numpy.mean(train1_features[:,0])
mean_std1 = numpy.mean(train1_features[:,1])
# Generate Variance of feature1(mean) and feature2 (standard deviation) for digit0 for the training dataset
var_mean0 = numpy.var(train0_features[:,0]) 
var_std0 = numpy.var(train0_features[:,1])
# Generate Variance of feature1(mean) and feature2 (standard deviation) for digit1 for the training dataset
var_mean1 = numpy.var(train1_features[:,0])
var_std1 = numpy.var(train1_features[:,1]) 


# Code for keeping mean and variance of features in neat dataframe
result1 = numpy.zeros((8,2))
result1 = pandas.DataFrame(result1, columns=['Components', 'Values'])

result1['Components'] = ('meanFeature1_0', 'varianceFeature1_0', 'meanFeature2_0', 'varianceFeature2_0',
               'meanFeature1_1', 'varianceFeature1_1', 'meanFeature2_1', 'varianceFeature2_1')
result1['Values'][0] = mean_mean0
result1['Values'][1] = var_mean0 
result1['Values'][2] = mean_std0 
result1['Values'][3] = var_std0 
result1['Values'][4] = mean_mean1
result1['Values'][5] = var_mean1
result1['Values'][6] = mean_std1
result1['Values'][7] = var_std1 
result1


# TASK 3
# 
# Bayes Theorem can be stated as:
# 
# Posterior = Likelihood * Prior / Evidence
# 
# Or
# 
# P(y|x) = P(x|y) * P(y) / P(x)
# 
# since P(x) is a constant, 
# 
# P(y|x) is proportional to P(x|y) * P(y)
# 
# 
# In this assignment a class of 0 or 1 is to be assigned based on two extracted features (mean and standard deviation of image pixels. The prior probabilities P(y=0) and P(y=0) are already assumed to be the same (0.5). 
# 
# Usually to create a classifier model, the probability of given set of inputs for all possible values of the class variable y is found and the output with maximum probability is selected. This can be expressed mathematically as:
# 
# y[hart] = argmax{P(x|y) * P(y)}
# 
# Since the prior for both classes (0 and 1)are same, then it can simply be ignored, therefore:
# 
# y[hart] = argmax{P(x|y)}
# 
# 
# 

# In[62]:


# Function for calculating P(x|y)
def calculate_prob_ygivenX(x1, x2, mean_mean, mean_std, variance_mean, variance_std):
    exponent_mean = math.exp(-((x1 - mean_mean)**2 / (2 * variance_mean )))
    exponent_std = math.exp(-((x2 - mean_std)**2 / (2 * variance_std )))
    gaussian_mean = 1 / (math.sqrt(2 * pi * variance_mean)) * exponent_mean
    gaussian_std = 1 / (math.sqrt(2 * pi * variance_std)) * exponent_std
    return gaussian_mean * gaussian_std
                    

# Function for calculating individual row of a test dataset either as digit 0 or digit 1.
def argmax_y(x1, x2):
    y0givenX = calculate_prob_ygivenX(x1, x2, mean_mean0, mean_std0, var_mean0, var_std0)
    y1givenX = calculate_prob_ygivenX(x1, x2, mean_mean1, mean_std1, var_mean1, var_std1)
    return numpy.argmax([y0givenX, y1givenX])
                                        
                    
# Function predicting new labels for test dataset                 
def testclass_y(dataset):
    class_y = []
    for i in range(len(dataset)):
        xdata = dataset[i]
        xmean = xdata[0]
        xstd = xdata[1]
        xclass = argmax_y(xmean, xstd)
        class_y.append(xclass)
    return class_y
                    
                    


# In[63]:


# Prediction of labels for digit 0 test dataset 
label0 = testclass_y(test0_features)
print(label0)


# In[64]:


# Prediction of labels for digit 1 test dataset   
label1 = testclass_y(test1_features) 
print(label1)


# TASK 4 
# Accuracy = number of labels correctly predicted / total number of labels in the test datasets

# In[65]:


# Accuracy for digit0 test dataset  
label0_accuracy = label0.count(0) / len(test0_features)
print(label0_accuracy)


# In[66]:


# Accuracy for digit1 test dataset 
label1_accuracy = label1.count(1) / len(test1_features)
print(label1_accuracy)


# TASK 5: cross check with sklearn

# In[67]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
Xtrainset = numpy.concatenate((train0_2d, train1_2d))
Xtrainset.shape
Ytrainset = numpy.zeros((10000,1))
Ytrainset[4999:10000,0] = 1
y_test0 = numpy.zeros((980,1))
y_test1 = numpy.zeros((1135,1))
y_test1[:,] = 1
y_test1
y_pred0 = gnb.fit(Xtrainset, Ytrainset).predict(test0_2d)
#print("Number of mislabeled points out of a total %d points : %d"
#      % (test0_2d.shape[0], (y_test0 != y_pred).sum()))
y_pred0.shape
x0 = numpy.count_nonzero(y_pred0)
accuracy0 = x0/len(y_test0)

y_pred1 = gnb.fit(Xtrainset, Ytrainset).predict(test1_2d)
y_pred1.shape
x1 = numpy.count_nonzero(y_pred1)
accuracy1 = x1/len(y_test1)

print(accuracy0, accuracy1)


# In[68]:



# Alternative function that switch the possitions of the probabilities, so that class 1 is favoured
def argmax_y_alt(x1, x2):
    y0givenX = calculate_prob_ygivenX(x1, x2, mean_mean0, mean_std0, var_mean0, var_std0)
    y1givenX = calculate_prob_ygivenX(x1, x2, mean_mean1, mean_std1, var_mean1, var_std1)
    return numpy.argmax([y1givenX, y0givenX])


def testclass_y_alt(dataset):
    class_y = []
    for i in range(len(dataset)):
        xdata = dataset[i]
        xmean = xdata[0]
        xstd = xdata[1]
        xclass = argmax_y_alt(xmean, xstd)
        class_y.append(xclass)
    return class_y
                    
# Prediction of labels for digit 0 test dataset using the alternative functions
label0_alt = testclass_y_alt(test0_features)
label0_alt_accuracy = label0_alt.count(0) / len(test0_features)

# Print alternative accuracy followed by predicted labels
print(label0_alt_accuracy, label0_alt)
                   


# References:
# 1. Lecture videos
# 2. https://www.geeksforgeeks.org/naive-bayes-classifiers/
# 3. https://scikit-learn.org/stable/modules/naive_bayes.html
