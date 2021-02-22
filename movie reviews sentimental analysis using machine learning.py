#!/usr/bin/env python
# coding: utf-8

# First import all neccessary libraries :which are required for text classification and machine learning libraries

# In[ ]:


##Movie Reviews Sentiment Analysis -Binary Classification with Machine Learning


# positive :good job,great work
# neutral:need to be improved
# negative:bad or worst experience etc

#  In this Machine Learning Project, we’ll build binary classification that puts movie reviews texts into one of two categories — negative or positive sentiment. We’re going to have a brief look at the Bayes theorem and relax its requirements using the Naive assumption.

# In[4]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re # for regexpressions
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score
import pickle
import os
os.getcwd()
os.chdir("E:\RESUME PROJECTS DATA SCIENCE")


# In[6]:


data = pd.read_csv('IMDB-Dataset.csv')


# In[7]:


data.head()


# In[8]:


data.shape


# In[9]:


data.info()


# In[10]:


data.isnull().sum()


# In[ ]:


##No null values, Label encode sentiment to 1(positive) and 0(negative)


# In[16]:


data.sentiment.replace('positive',1,inplace=True)


# In[17]:


data.sentiment.replace('negative',0,inplace=True)


# In[18]:


data.head(10)


# In[19]:


data['sentiment'].value_counts()


# In[20]:


data['review'][0]


# STEPS TO CLEAN THE REVIEWS :
# Remove HTML tags
# Remove special characters
# Convert everything to lowercase
# Remove stopwords
# Stemming
# 
# 

# In[ ]:


#1.removal of hash tags   Regex rule : ‘<.*?>’


# In[21]:


def clean(text):
    cleaned = re.compile(r'<.*?>')
    return re.sub(cleaned,'',text)

data.review = data.review.apply(clean)
data.review[0]


# 2. Remove special characters
# 

# In[22]:


def is_special(text):
    rem = ''
    for i in text:
        if i.isalnum():
            rem = rem + i
        else:
            rem = rem + ' '
    return rem


# In[23]:


data.review = data.review.apply(is_special)
data.review[0]


# 3. Convert everything to lowercase
# 

# In[24]:


def to_lower(text):
    return text.lower()


# In[25]:


data.review = data.review.apply(to_lower)
data.review[0]


# 4. Remove stopwords
# 

# In[26]:


def rem_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return [w for w in words if w not in stop_words]


# In[27]:


data.review = data.review.apply(rem_stopwords)
data.review[0]


# 5. Stem the words
# 

# In[28]:


def stem_txt(text):
    ss = SnowballStemmer('english')
    return " ".join([ss.stem(w) for w in text])

data.review = data.review.apply(stem_txt)
data.review[0]


# In[29]:


data.head()


# CREATING THE MODEL
# 

# In[30]:


#1. Creating Bag Of Words (BOW)
X = np.array(data.iloc[:,0].values)
y = np.array(data.sentiment.values)
cv = CountVectorizer(max_features = 1000)
X = cv.fit_transform(data.review).toarray()
print("X.shape = ",X.shape)
print("y.shape = ",y.shape)


# In[31]:


print(X)


# In[32]:


print(y)


# In[ ]:


#2. Train test split


# In[33]:


trainx,testx,trainy,testy = train_test_split(X,y,test_size=0.2,random_state=9)


# In[34]:


print("Train shapes : X = {}, y = {}".format(trainx.shape,trainy.shape))


# In[35]:


print("Test shapes : X = {}, y = {}".format(testx.shape,testy.shape))


# In[ ]:


#3. Defining the models and Training them


# In[37]:


gnb,mnb,bnb = GaussianNB(),MultinomialNB(alpha=1.0,fit_prior=True),BernoulliNB(alpha=1.0,fit_prior=True)
gnb.fit(trainx,trainy)
mnb.fit(trainx,trainy)
bnb.fit(trainx,trainy)


# In[ ]:


#4. Prediction and accuracy metrics to choose best model


# In[38]:


ypg = gnb.predict(testx)
ypm = mnb.predict(testx)
ypb = bnb.predict(testx)


# In[39]:



print("Gaussian = ",accuracy_score(testy,ypg))
print("Multinomial = ",accuracy_score(testy,ypm))
print("Bernoulli = ",accuracy_score(testy,ypb))


# In[49]:


from sklearn.metrics import confusion_matrix


# In[50]:


cf=confusion_matrix(testy,ypg)


# In[52]:


cf1=confusion_matrix(testy,ypm)


# In[53]:


print(cf1)


# In[54]:


cf2=confusion_matrix(testy,ypb)


# In[55]:


print(cf2)


# In[51]:


print(cf)


# In[40]:


pickle.dump(bnb,open('model1.pkl','wb'))


# In[42]:


rev =  """Terrible. Complete trash. Brainless tripe. Insulting to anyone who isn't an 8 year old fan boy. Im actually pretty disgusted that this movie is making the money it is - what does it say about the people who brainlessly hand over the hard earned cash to be 'entertained' in this fashion and then come here to leave a positive 8.8 review?? Oh yes, they are morons. Its the only sensible conclusion to draw. How anyone can rate this movie amongst the pantheon of great titles is beyond me.

So trying to find something constructive to say about this title is hard...I enjoyed Iron Man? Tony Stark is an inspirational character in his own movies but here he is a pale shadow of that...About the only 'hook' this movie had into me was wondering when and if Iron Man would knock Captain America out...Oh how I wished he had :( What were these other characters anyways? Useless, bickering idiots who really couldn't organise happy times in a brewery. The film was a chaotic mish mash of action elements and failed 'set pieces'...

I found the villain to be quite amusing.

And now I give up. This movie is not robbing any more of my time but I felt I ought to contribute to rest"""


# In[43]:


f1 = clean(rev)
f2 = is_special(f1)
f3 = to_lower(f2)
f4 = rem_stopwords(f3)
f5 = stem_txt(f4)


# In[44]:


bow,words = [],word_tokenize(f5)
for word in words:
    bow.append(words.count(word))


# In[45]:


word_dict = cv.vocabulary_


# In[46]:


pickle.dump(word_dict,open('bow.pkl','wb'))


# In[48]:


inp = []
for i in word_dict:
    inp.append(f5.count(i[0]))
y_pred = bnb.predict(np.array(inp).reshape(1,1000))


# In[ ]:




