import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
words=nltk.download('stopwords')
stopwords.words('english')
dataset = pd.read_csv("train.csv")
dataset.head()
dataset.shape
dataset.isnull().sum()
dataset.fillna('', inplace=True)
dataset.isnull().sum()
dataset['content']=dataset['title']+dataset['author']
dataset.shape
dataset['content']
x=dataset.drop(columns='label', axis=1)
y=dataset['label']
print(x)
print(y)
port_stemmer=PorterStemmer()
def stemming(content):
  stemmed_content=re.sub('[^a-zA-Z]',' ',content)
  stemmed_content=stemmed_content.lower()
  stemmed_content=stemmed_content.split()
  stemmed_content=[port_stemmer.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
  stemmed_content=' '.join(stemmed_content)
  return stemmed_content
dataset['content']=dataset['content'].apply(stemming)
print(dataset['content'])
x=dataset['content'].values
y=dataset['label'].values 
vectorizer=TfidfVectorizer()
vectorizer.fit(x)
x=vectorizer.transform(x)
print (x)
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.2, stratify=y, random_state=2)
x.shape, x_train.shape, x_test.shape
model=LogisticRegression()
model.fit(x_train, y_train)
prediction=model.predict(x_train)
accuracy=accuracy_score(prediction, y_train)
print("Accuracy of trained Dataset --> ", round(np.multiply(accuracy, 100),2), "%")
model.fit(x_test, y_test)
prediction=model.predict(x_test)
accuracy=accuracy_score(prediction, y_test)
print("Accuracy of test Dataset --> ", round(np.multiply(accuracy, 100),2), "%")
x=x_test[4159] #test data mein total 4159 values lii gayi hai toh isse jyada ka testing results nhi dega
prediction=model.predict(x)
if (prediction[0]==0):
  print('The news is Real')
else:
  print('The news is Fake')
