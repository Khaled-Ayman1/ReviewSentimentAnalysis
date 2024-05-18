import os
import joblib
from sklearn.metrics import accuracy_score

import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from transformers import pipeline

def preprocessing(text):
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

def load_test(folder_path):
    reviews = []
    for filename in os.listdir(folder_path):
        with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
            review = file.read()
            reviews.append(review)
    return reviews

vectorizer = joblib.load('models/dependencies/vectorizer')

folder_path_positive = 'dataset/pos_test'
folder_path_negative = 'dataset/neg_test'

test_data_positive = load_test(folder_path_positive)
test_data_negative = load_test(folder_path_negative)

X = test_data_positive + test_data_negative
Y = ['positive'] * len(test_data_positive) + ['negative'] * len(test_data_negative)

X_test = [' '.join(preprocessing(test_review)) for test_review in X]

vectorized_X = vectorizer.transform(X_test)

lin_SVM = joblib.load('models/lin_SVM')
SVM_pred = lin_SVM.predict(vectorized_X)
print("Linear SVM Accuracy: ", accuracy_score(Y, SVM_pred))

CV_SVM = joblib.load('models/CV_SVM')
CV_pred = CV_SVM.predict(vectorized_X)
print("Grid Search SVM Accuracy: ", accuracy_score(Y, CV_pred))

nlp = pipeline('sentiment-analysis')
X_test_truncated = [review[:512] for review in X_test]

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
accuracy_bert = accuracy_score(Y, y_pred_bert)
print("BERT Accuracy:", accuracy_bert)
