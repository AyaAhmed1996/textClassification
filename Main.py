import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB ,BernoulliNB ,GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVR,SVC
import string

# the dataset
news_df = pd.read_csv("uci-news-aggregator.csv  ", sep = ",")

# part from the dataset for SVC
#news_df = pd.read_csv("uci-news-aggregator-SVC.csv  ", sep = ",")


# converting strings to int to be more easy , converting to lowercase , removing punctuation , specifying data for testing
news_df['CATEGORY'] = news_df.CATEGORY.map({ 'b': 1, 't': 2, 'e': 3, 'm': 4 })
news_df['TITLE'] = news_df.TITLE.map(lambda x: x.lower().translate(str.maketrans('','', string.punctuation)))
X_train, X_test, y_train, y_test = train_test_split(news_df['TITLE'],news_df['CATEGORY'],test_size=0.3,random_state =42)


# removing stopwords and transforming data
#tf_vector = TfidfVectorizer(stop_words = 'english',max_df=20,min_df=5)
tf_vector = TfidfVectorizer(stop_words = 'english')
training_data = tf_vector.fit_transform(X_train.values)
testing_data = tf_vector.transform(X_test.values)


# testing classifiers to compare accuracy
#naive_bayes = SVC(C=13, cache_size=200, class_weight=None, decision_function_shape='ovr',
#                 degree=3, gamma= 'auto', kernel='rbf',max_iter=-1, probability=False,
#                 random_state=0, shrinking=True,tol=0.01, verbose=0)
#naive_bayes = DecisionTreeClassifier()
#naive_bayes = BernoulliNB()
naive_bayes = MultinomialNB()
hist=naive_bayes.fit(training_data, y_train)


y_pred = naive_bayes.predict(testing_data)
#print(y_pred)


# print result
print("Accuracy Percent:", int(accuracy_score(y_test,y_pred)*100),"%")
print("Accuracy Score:", accuracy_score(y_test,y_pred))

