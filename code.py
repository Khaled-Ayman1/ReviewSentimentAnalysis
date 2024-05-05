import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix


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


#Model training
clf = SVC(kernel='linear')
clf.fit(X_train_vec, y_train)
accuracy_train = clf.score(X_train_vec, y_train)

#Model testing
y_pred = clf.predict(X_test_vec)
accuracy_test = accuracy_score(y_test, y_pred)
model_classification_report = classification_report(y_test, y_pred)
print("Accuracy:", accuracy_test)
print("Classification Report:\n", model_classification_report)

#Plot the results
#Still to be done

#Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['negative', 'positive'], yticklabels=['negative', 'positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()