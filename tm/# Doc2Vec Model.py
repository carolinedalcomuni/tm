# Doc2Vec Model
def train_doc2vec_model(train_reviews, train_labels, vector_size=150):
    tagged_data = [TaggedDocument(words=preprocess_text(review), tags=[label]) for review, label in zip(train_reviews, train_labels)]
    model = Doc2Vec(tagged_data, vector_size=vector_size)
    return model

doc2vec_model = train_doc2vec_model(train['review'], train['sentiment'])
train_doc2vec_vectors = [doc2vec_model.infer_vector(preprocess_text(review)) for review in train['review']]
test_doc2vec_vectors = [doc2vec_model.infer_vector(preprocess_text(review)) for review in test['review']]
doc2vec_dt_metrics = evaluate_model(train_dt_model(train_doc2vec_vectors, train['sentiment']), test_doc2vec_vectors, test['sentiment'])
print("Doc2Vec Decision Tree Metrics:", doc2vec_dt_metrics)


# Execute as funções para obter as métricas - texto pré processado com Stemming
vectorizer4 = TfidfVectorizer(tokenizer=preprocess_text4, max_features=10000)
train_vectors, test_vectors = vectorize_text(train['review'], test['review'], vectorizer4)
dt_metrics4, dt_model4 = train_and_evaluate_model(train_vectors, test_vectors, train_labels=train['sentiment'], test_labels=test['sentiment'], model_type='dt')
lr_metrics4, lr_model4 = train_and_evaluate_model(train_vectors, test_vectors, train_labels=train['sentiment'], test_labels=test['sentiment'], model_type='lr')
svm_metrics4, svm_model4 = train_and_evaluate_model(train_vectors, test_vectors, train_labels=train['sentiment'], test_labels=test['sentiment'], model_type='svm_linear')


# Imprima as métricas corretamente
print_metrics(dt_metrics4, "Decision Tree Metrics:")
print_metrics(lr_metrics4, "Logistic Regression Metrics:")
print_metrics(svm_metrics4, "SVM Metrics (Linear Kernel):")

# Carregar o tokenizador e o modelo BERT para classificação de sequências
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Exemplo de texto para classificação
text = test.iloc[0, 1] # Acessa a primeira linha (índice 0) e segunda coluna (índice 1)

# Tokenizar o texto e preparar para entrada no modelo
inputs = tokenizer(text, return_tensors='pt')

# Passar os inputs pelo modelo para obter as previsões
outputs = model(**inputs)

# Obter as probabilidades de cada classe
probs = torch.softmax(outputs.logits, dim=1)
predicted_class = torch.argmax(probs, dim=1)

# Mostrar as previsões
print(probs)
print(predicted_class)