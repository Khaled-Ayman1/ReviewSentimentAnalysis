import os
import joblib

import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from transformers import pipeline
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
import seaborn as sns


# Download NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download("stopwords")


# Preprocessing and Lemmatization
def preprocess_text(text):

    #Tokenization
    words = nltk.word_tokenize(text)

    stop_words = set(stopwords.words("english"))
    tokens = []

    for word in words:
        if word.casefold() not in stop_words:
            tokens.append(word)

    # POS tagging
    pos_tags = nltk.pos_tag(tokens)

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_text = []
    for word, pos in pos_tags:
        if pos.startswith('V'):  #Verb
            pos = wordnet.VERB
        elif pos.startswith('J'):  #Adjective
            pos = wordnet.ADJ
        elif pos.startswith('R'):  #Adverb
            pos = wordnet.ADV
        else:
            pos = wordnet.NOUN  #Default value noun
        lemmatized_text.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_text

#Read from files
def load_reviews_from_folder(folder_path):
    reviews = []
    for filename in os.listdir(folder_path):
        with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
            review = file.read()
            reviews.append(review)
    return reviews

#datasets
positive_reviews = load_reviews_from_folder('dataset/pos')
negative_reviews = load_reviews_from_folder('dataset/neg')

#Add all pos and neg reviews
X = positive_reviews + negative_reviews
#Add labels
y = ['positive'] * len(positive_reviews) + ['negative'] * len(negative_reviews)


#Apply preprocessing to each review in all the features
X_preprocessed = [' '.join(preprocess_text(review)) for review in X]


# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=44)

#IDF
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

joblib.dump(vectorizer, 'models/dependencies/vectorizer')

#Linear Model training
clf = SVC(kernel='linear')
clf.fit(X_train_vec, y_train)
accuracy_train = clf.score(X_train_vec, y_train)

#Linear Model testing
y_pred_linear = clf.predict(X_test_vec)
accuracy_test = accuracy_score(y_test, y_pred_linear)
model_classification_report = classification_report(y_test, y_pred_linear)
print("Linear SVM Accuracy:", accuracy_test)
print("Classification Report:\n", model_classification_report)

joblib.dump(clf, 'models/lin_SVM')

#SVM with gridsearch training
parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
svc = SVC()
clf = GridSearchCV(svc, parameters)
clf.fit(X_train_vec, y_train)
accuracy_train = clf.score(X_train_vec, y_train)
#print("Best parameters:", clf.best_params)

#SVM with gridsearch testing
y_pred_grid = clf.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred_grid)
print("Grid Search SVM Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred_grid))

joblib.dump(clf, 'models/CV_SVM')


#BERT
nlp = pipeline('sentiment-analysis')
# Truncate each review in X_test to a maximum length of 512 tokens
X_test_truncated = [review[:512] for review in X_test]
# Process input reviews in batches and predict sentiment labels
batch_size = 8  # Adjust batch size as needed
y_pred_bert = []
for i in range(0, len(X_test_truncated), batch_size):
    batch_reviews = X_test_truncated[i:i+batch_size]
    batch_outputs = nlp(batch_reviews)
    batch_predictions = [output['label'] for output in batch_outputs]
    y_pred_bert.extend(batch_predictions)
# Convert sentiment labels to 'positive' or 'negative'
y_pred_bert = ['positive' if label == 'POSITIVE' else 'negative' for label in y_pred_bert]
# Evaluate accuracy and print results
accuracy_bert = accuracy_score(y_test, y_pred_bert)
print("BERT Accuracy:", accuracy_bert)
print("Classification Report:\n", classification_report(y_test, y_pred_bert))


#Plot confusion matrix
#SVM linear
cm = confusion_matrix(y_test, y_pred_linear)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['negative', 'positive'], yticklabels=['negative', 'positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('SVM linear Confusion Matrix')
plt.show()

#SVM CV
cm = confusion_matrix(y_test, y_pred_grid)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['negative', 'positive'], yticklabels=['negative', 'positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('SVM GridSearch Confusion Matrix')
plt.show()

#Bert
cm = confusion_matrix(y_test, y_pred_bert)
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)  # Adjust font scale if needed
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['negative', 'positive'], yticklabels=['negative', 'positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Bert Confusion Matrix')
plt.show()