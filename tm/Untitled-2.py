# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import classification_report, accuracy_score
from nltk.tokenize import word_tokenize
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import cross_val_score, KFold
from textblob import TextBlob
from sklearn.svm import SVC
import time

# %%
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

path_to_resources = 'resources/'

# Load data
train = pd.read_csv(path_to_resources + "amazon_reviews_train.csv")
test = pd.read_csv(path_to_resources + "amazon_reviews_test.csv")

# %% [markdown]
# **1.1 Dados**

# %%
# Exploratory Data Analysis
def plot_sentiment_distribution(data, title):
    sentiment_counts = data['sentiment'].value_counts()
    labels = sentiment_counts.index.tolist()
    counts = sentiment_counts.values.tolist()

    plt.figure()
    plt.pie(counts, labels=labels, autopct='%1.2f%%', colors=['#129912', '#CF110A'])
    plt.title(title)
    plt.show()

plot_sentiment_distribution(train, 'Distribuição das etiquetas de sentimento do conjunto treino')
plot_sentiment_distribution(test, 'Distribuição das etiquetas de sentimento do conjunto teste')

# %%
reviews_teste=test["review"].tolist()
sentimento_teste=test["sentiment"].tolist()

reviews_treino=train["review"].tolist()
sentimento_treino=train["sentiment"].tolist()

print(f"Total de reviews:{len(reviews_treino)+len(reviews_teste)}")
print("Percentagem do conjunto de treino: {:.3f}".format(len(reviews_treino)/(len(reviews_treino)+len(reviews_teste))*100))

# %% [markdown]
# **1.2 Definição de um baseline usando ferramentas já existentes**

# %%
def sentiment_label(score):
    if score > 0:
        return 'positive'
    else:
        return 'negative'
    
def add_to_confusion_matrix(confusion_matrix, true_sentiment, predicted_sentiment):
    if true_sentiment == "positive":
        if true_sentiment == predicted_sentiment:
            confusion_matrix["TP"] += 1
        else:
            confusion_matrix["FN"] += 1
    else:
        if true_sentiment == predicted_sentiment:
            confusion_matrix["TN"] += 1
        else:
            confusion_matrix["FP"] += 1
            
def get_metrics(confusion_matrix):
    accuracy = round((confusion_matrix["TP"] + confusion_matrix["TN"]) / (confusion_matrix["TP"] + confusion_matrix["FN"] + confusion_matrix["FP"] + confusion_matrix["TN"]), 4)
    precision = round((confusion_matrix["TP"]) / (confusion_matrix["TP"] + confusion_matrix["FP"]), 4)
    recall = round((confusion_matrix["TP"]) / (confusion_matrix["TP"] + confusion_matrix["FN"]), 4)
    f1 = round((2 * precision * recall) / (precision + recall),4)
    return accuracy, precision, recall, f1


# %% [markdown]
# **TextBlob**

# %%
def textblob_metrics(reviews, labels):
    confusion_matrix = {"TP" : 0, "FN" : 0, "FP" : 0, "TN" : 0}
    results = {}
    start_time = time.time()
    for i in range(len(reviews)):
        review = reviews[i]
        true_sentiment = labels[i]
        score = round(TextBlob(review).sentiment.polarity, 3)
        predicted_sentiment = sentiment_label(score)
        add_to_confusion_matrix(confusion_matrix, true_sentiment, predicted_sentiment)
    accuracy, precision, recall, f1 = get_metrics(confusion_matrix)
    results["textblob_metrics"] = [accuracy, precision, recall, f1,round(((time.time()-start_time) / 60),2)]
    return pd.DataFrame.from_dict(results, orient='index',columns=["Accuracy","Precision","Recall","F1","Time"])

textblob_metrics_result = textblob_metrics(reviews_teste, sentimento_teste)
print(textblob_metrics_result)

# %% [markdown]
# **Vader Sentiment**

# %%
# Baseline Metrics using Vader Sentiment

sa = SentimentIntensityAnalyzer()

texts = test['review'].tolist()
for text in texts:
    scores = sa.polarity_scores(text)
    print('Review:', text)
    print('Positive:', scores['pos'])
    print('Negative:', scores['neg'])
    print('Compound Sentiment Score:', scores['compound'])
    print()
df = pd.read_csv(path_to_resources + 'amazon_reviews_test.csv')

# Predict sentiments
df['predicted_label'] = df['review'].apply(lambda x: sentiment_label(sa.polarity_scores(x)['compound']))

# Calculate and print classification report and accuracy
print(classification_report(df['sentiment'], df['predicted_label']))
accuracy = accuracy_score(df['sentiment'], df['predicted_label'])
print('Accuracy:', accuracy)

# %%
def print_metrics(metrics, model_name):
    accuracy, precision, recall, f1 = metrics
    print(f"{model_name} Metrics:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

# %% [markdown]
# 
# **1.3 Preparação de dados e aplicação de um léxico de sentimentos**

# %% [markdown]
# **NEGATION HANDLING**

# %%
# Text Processing

def preprocess_text1(text, lower=True, remove_stopwords=True, handle_negation=True):
    # Lowercasing
    if lower:
        text = text.lower()

    # Tokenization 
    tokens = word_tokenize(text)

    # Removing stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words("english"))
        tokens = [word for word in tokens if word not in stop_words]

    # Negation
    if handle_negation:
        negation_words = ["not", "no", "never", "none", "neither", "nor", "didn't like", "did not", "hate"]
        negated = False
        for i in range(len(tokens)):
            if tokens[i] in negation_words:
                negated = not negated
            elif negated and tokens[i] not in [",", ".", "!", "?"]:
                tokens[i] = "NOT_" + tokens[i]
    return tokens

# %%
def get_lexicon_score(review, lexicon_dict):
    tokens = preprocess_text1(review)
    positive_score = sum(lexicon_dict.get(token, {}).get('Positive', 0) for token in tokens)
    negative_score = sum(lexicon_dict.get(token, {}).get('Negative', 0) for token in tokens)
    return "positive" if positive_score > negative_score else "negative"

def calculate_metrics(labels, predicted_sentiments):
    accuracy = accuracy_score(labels, predicted_sentiments)
    precision = precision_score(labels, predicted_sentiments, average='binary', pos_label='positive')
    recall = recall_score(labels, predicted_sentiments, average='binary', pos_label='positive')
    f1 = f1_score(labels, predicted_sentiments, average='binary', pos_label='positive')
    
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

def print_metrics(metrics, title):
    print(f"{title}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")

# %%
lexicon = pd.read_csv(path_to_resources + "NCR-lexicon.csv")
lexicon_dict = lexicon.set_index('English').to_dict('index')

def get_lexicon_metrics(reviews, labels, lexicon_dict):
    predicted_sentiments = [get_lexicon_score(review, lexicon_dict) for review in reviews]
    metrics = {
        'Accuracy': accuracy_score(labels, predicted_sentiments),
        'Precision': precision_score(labels, predicted_sentiments, pos_label="positive"),
        'Recall': recall_score(labels, predicted_sentiments, pos_label="positive"),
        'F1 Score': f1_score(labels, predicted_sentiments, pos_label="positive")
    }
    return metrics

lexicon_metrics = get_lexicon_metrics(test['review'], test['sentiment'], lexicon_dict)
print_metrics(lexicon_metrics, "Lexicon Metrics")

# %% [markdown]
# **WHITHOUT NEGATION HANDLING**

# %%
# Text Processing 

def preprocess_text2(text, lower=True, remove_stopwords=True):
    # Lowercasing
    if lower:
        text = text.lower()

    # Tokenization 
    tokens = word_tokenize(text)

    # Removing stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words("english"))
        tokens = [word for word in tokens if word not in stop_words]

    return tokens

# %%
def get_lexicon_score(review, lexicon_dict):
    tokens = preprocess_text2(review)
    positive_score = sum(lexicon_dict.get(token, {}).get('Positive', 0) for token in tokens)
    negative_score = sum(lexicon_dict.get(token, {}).get('Negative', 0) for token in tokens)
    return "positive" if positive_score > negative_score else "negative"

def calculate_metrics(labels, predicted_sentiments):
    accuracy = accuracy_score(labels, predicted_sentiments)
    precision = precision_score(labels, predicted_sentiments, average='binary', pos_label='positive')
    recall = recall_score(labels, predicted_sentiments, average='binary', pos_label='positive')
    f1 = f1_score(labels, predicted_sentiments, average='binary', pos_label='positive')
    
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

def print_metrics(metrics, title):
    print(f"{title}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")

# %%
lexicon = pd.read_csv(path_to_resources + "NCR-lexicon.csv")
lexicon_dict = lexicon.set_index('English').to_dict('index')

def get_lexicon_metrics(reviews, labels, lexicon_dict):
    predicted_sentiments = [get_lexicon_score(review, lexicon_dict) for review in reviews]
    metrics = {
        'Accuracy': accuracy_score(labels, predicted_sentiments),
        'Precision': precision_score(labels, predicted_sentiments, pos_label="positive"),
        'Recall': recall_score(labels, predicted_sentiments, pos_label="positive"),
        'F1 Score': f1_score(labels, predicted_sentiments, pos_label="positive")
    }
    return metrics

lexicon_metrics = get_lexicon_metrics(test['review'], test['sentiment'], lexicon_dict)
print_metrics(lexicon_metrics, "Lexicon Metrics")

# %% [markdown]
# **LEMMATIZATION**

# %%
# Text Preprocessing

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocess_text3(text, lower=True, remove_stopwords=True, lemmatization=False):
    # Lowercasing
    if lower:
        text = text.lower()

    # Tokenization 
    tokens = word_tokenize(text)

    # Removing stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words("english"))
        tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    if lemmatization:
        lemmatizer = WordNetLemmatizer()
        pos_tagged = nltk.pos_tag(tokens)
        tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tagged]
        
    return tokens

# %%
def get_lexicon_score(review, lexicon_dict):
    tokens = preprocess_text3(review)
    positive_score = sum(lexicon_dict.get(token, {}).get('Positive', 0) for token in tokens)
    negative_score = sum(lexicon_dict.get(token, {}).get('Negative', 0) for token in tokens)
    return "positive" if positive_score > negative_score else "negative"

def calculate_metrics(labels, predicted_sentiments):
    accuracy = accuracy_score(labels, predicted_sentiments)
    precision = precision_score(labels, predicted_sentiments, average='binary', pos_label='positive')
    recall = recall_score(labels, predicted_sentiments, average='binary', pos_label='positive')
    f1 = f1_score(labels, predicted_sentiments, average='binary', pos_label='positive')
    
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

def print_metrics(metrics, title):
    print(f"{title}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")

# %%
lexicon = pd.read_csv(path_to_resources + "NCR-lexicon.csv")
lexicon_dict = lexicon.set_index('English').to_dict('index')

def get_lexicon_metrics(reviews, labels, lexicon_dict):
    predicted_sentiments = [get_lexicon_score(review, lexicon_dict) for review in reviews]
    metrics = {
        'Accuracy': accuracy_score(labels, predicted_sentiments),
        'Precision': precision_score(labels, predicted_sentiments, pos_label="positive"),
        'Recall': recall_score(labels, predicted_sentiments, pos_label="positive"),
        'F1 Score': f1_score(labels, predicted_sentiments, pos_label="positive")
    }
    return metrics

lexicon_metrics = get_lexicon_metrics(test['review'], test['sentiment'], lexicon_dict)
print_metrics(lexicon_metrics, "Lexicon Metrics")


# %% [markdown]
# **STEMMING**

# %%
# Stemmizar o léxico

def prepare_lexicon(lexicon_path):
    lexicon = pd.read_csv(lexicon_path)
    stemmer = PorterStemmer()
    # Certifique-se de que todos os valores são strings
    lexicon['English'] = lexicon['English'].fillna('')
    lexicon['Stemmed'] = lexicon['English'].apply(lambda x: stemmer.stem(x.lower()))
    
    lexicon_aggregated = lexicon.groupby('Stemmed').agg({
        'Positive': 'sum',  
        'Negative': 'sum' 
    }).reset_index()

    return lexicon_aggregated.set_index('Stemmed').to_dict('index')

# Caminho para o arquivo do léxico
lexicon_path = path_to_resources + 'NCR-lexicon.csv'

# Preparar o léxico
lexicon_dict = prepare_lexicon(lexicon_path)

# %%
# Text Preprocessing
def preprocess_text4(text, lower=True, remove_stopwords=True, stemming=True):
    if lower:
        text = text.lower()
    
    tokens = word_tokenize(text)
    
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
    
    if stemming:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]
    
    return tokens

# %%
def get_lexicon_score(review, lexicon_dict):
    tokens = preprocess_text4(review)
    positive_score = sum(lexicon_dict.get(token, {}).get('Positive', 0) for token in tokens)
    negative_score = sum(lexicon_dict.get(token, {}).get('Negative', 0) for token in tokens)
    return "positive" if positive_score > negative_score else "negative"

def calculate_metrics(labels, predicted_sentiments):
    accuracy = accuracy_score(labels, predicted_sentiments)
    precision = precision_score(labels, predicted_sentiments, average='binary', pos_label='positive')
    recall = recall_score(labels, predicted_sentiments, average='binary', pos_label='positive')
    f1 = f1_score(labels, predicted_sentiments, average='binary', pos_label='positive')
    
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

def print_metrics(metrics, title):
    print(f"{title}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")

# %%
def get_lexicon_metrics(reviews, labels, lexicon_dict):
    predicted_sentiments = [get_lexicon_score(review, lexicon_dict) for review in reviews]
    metrics = {
        'Accuracy': accuracy_score(labels, predicted_sentiments),
        'Precision': precision_score(labels, predicted_sentiments, pos_label="positive"),
        'Recall': recall_score(labels, predicted_sentiments, pos_label="positive"),
        'F1 Score': f1_score(labels, predicted_sentiments, pos_label="positive")
    }
    return metrics

lexicon_metrics = get_lexicon_metrics(test['review'], test['sentiment'], lexicon_dict)
print_metrics(lexicon_metrics, "Lexicon Metrics")

# %% [markdown]
# **4. Treino de um modelo (aprendizagem automática)**

# %%
# Text Vectorization - converter texto em representações numéricas (vetores) adequadas para modelos de aprendizado de máquina.
def vectorize_text(train_reviews, test_reviews, vectorizer):
    train_vectors = vectorizer.fit_transform(train_reviews)
    test_vectors = vectorizer.transform(test_reviews)
    return train_vectors, test_vectors

# %%
# Modelos

# Decision Tree Model
def train_dt_model(train_vectors, train_labels):
    model = tree.DecisionTreeClassifier()
    model.fit(train_vectors, train_labels)
    return model

# Logistic Regression Model
def train_lr_model(train_vectors, train_labels):
    model = LogisticRegression(max_iter=300)
    model.fit(train_vectors, train_labels)
    return model

# SVM Model
def train_svm_model(train_vectors, train_labels, kernel='linear'):
    model = svm.SVC(kernel=kernel)
    model.fit(train_vectors, train_labels)
    return model

# %%
def evaluate_model(model, test_vectors, test_labels):
    predictions = model.predict(test_vectors)
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, average='weighted')
    recall = recall_score(test_labels, predictions, average='weighted')
    f1 = f1_score(test_labels, predictions, average='weighted')
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    return metrics

# %%
# Train and evaluate models
def train_and_evaluate_model(train_vectors, test_vectors, train_labels, test_labels, model_type):
    model = None
    if model_type == 'dt':
        model = DecisionTreeClassifier()
    elif model_type == 'lr':
        model = LogisticRegression()
    elif model_type == 'svm_linear':
        model = SVC(kernel='linear')
    else:
        raise ValueError("Invalid model type")
    
    model.fit(train_vectors, train_labels)
    metrics = evaluate_model(model, test_vectors, test_labels)
    return metrics, model

# %%
# Execute as funções para obter as métricas
vectorizer = TfidfVectorizer(tokenizer=preprocess_text, lowercase=True)
train_vectors, test_vectors = vectorize_text(train['review'], test['review'], vectorizer)
dt_metrics, dt_model = train_and_evaluate_model(train_vectors, test_vectors, train_labels=train['sentiment'], test_labels=test['sentiment'], model_type='dt')
lr_metrics, lr_model = train_and_evaluate_model(train_vectors, test_vectors, train_labels=train['sentiment'], test_labels=test['sentiment'], model_type='lr')
svm_metrics, svm_model = train_and_evaluate_model(train_vectors, test_vectors, train_labels=train['sentiment'], test_labels=test['sentiment'], model_type='svm_linear')

# Imprima as métricas corretamente
print_metrics(dt_metrics, "Decision Tree Metrics:")
print_metrics(lr_metrics, "Logistic Regression Metrics:")
print_metrics(svm_metrics, "SVM Metrics (Linear Kernel):")


# %%
# Cross-validation
def cross_validate(model, train_vectors, train_labels):
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, train_vectors, train_labels, cv=cv, scoring='accuracy')
    return np.mean(scores)

# Cross-validation para o modelo de regressão logística
svm_cv_score = cross_validate(svm_model, train_vectors, train['sentiment'])
cv_metrics = {'Accuracy': svm_cv_score, 'Precision': svm_cv_score, 'Recall': svm_cv_score, 'F1 Score': svm_cv_score}
print_metrics(cv_metrics, "SVM Cross-validation Score:")

# %% [markdown]
# **5.Utilização de transformadores para classificação**
# 
# Fonte: https://towardsdatascience.com/sentiment-analysis-with-pretrained-transformers-using-pytorch-420bbc1a48cd
# 
# Versão adaptada para o presente trabalho da cadeira de Text Mining

# %% [markdown]
# 


