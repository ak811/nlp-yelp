import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report

sns.set_style('white')

yelp = pd.read_csv('yelp.csv')

# the "text length" column: number of words in the text column
yelp['text length'] = yelp['text'].apply(len)

# using FacetGrid from seaborn
g = sns.FacetGrid(yelp, col='stars')
g.map(plt.hist, 'text length')

# a boxplot of text length for each star category
sns.boxplot(x='stars', y='text length', data=yelp, palette='rainbow')

# a countplot for the number of occurrences
sns.countplot(x='stars', data=yelp, palette='rainbow')

# get the mean values of the numerical columns and produce the dataframe
stars = yelp.groupby('stars').mean()
dataframe = stars.corr()

# create a heatmap based off the dataframe:**
sns.heatmap(dataframe, cmap='coolwarm', annot=True)

# classification: yelp_class dataframe for the 1 or 5 star reviews
yelp_class = yelp[(yelp.stars == 1) | (yelp.stars == 5)]

X = yelp_class['text']
y = yelp_class['stars']

cv = CountVectorizer()

X = cv.fit_transform(X)

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# training the model
nb = MultinomialNB()
nb.fit(X_train, y_train)

predictions = nb.predict(X_test)

print(confusion_matrix(y_test, predictions))
print('\n')
print(classification_report(y_test, predictions))

# using TF-IDF to create a pipeline
pipeline = Pipeline([
    ('bow', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB()),
])

X = yelp_class['text']
y = yelp_class['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
