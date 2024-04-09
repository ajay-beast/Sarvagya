import torch
import tensorflow as tf
from transformers import BertTokenizer, BertModel
import numpy as np

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to create user embeddings
def createUserEmbedding(session_query_list, session_response):
    session_embedding = []
    for quries, responses in zip(session_query_list, session_response):
        list_session_embedding = createListOfSessionEmbedding(quries, responses)
        concatinated_embedding = concatination(list_session_embedding)
        single_embedding = get_prediction(concatinated_embedding,'mse')

        session_embedding.append(single_embedding)
    
    concatinated_embedding = concatination(session_embedding)

    return get_prediction(concatinated_embedding,'mse')



def createQueryResponseEmbedding(query, response):
    inputs = tokenizer(query, response, return_tensors="pt", truncation=True)

    outputs = model(**inputs)

    last_hidden_states = outputs.last_hidden_state

    mean_embedding = torch.mean(last_hidden_states, dim=1)

    mean_embedding = mean_embedding.detach().numpy()

    return mean_embedding

def createListOfSessionEmbedding(queryes_session, responses_session):
    list_session_embedding = []
    for query, response in zip(queryes_session, responses_session):
        embedding = createQueryResponseEmbedding(query, response)
        list_session_embedding.append(embedding)

    return list_session_embedding

def concatination(list_session_embedding):
    list_of_embeddings = []

    for pair in list_session_embedding:
        list_of_embeddings.append(pair.flatten())
    
    X_train_data = np.concatenate(list_of_embeddings)
    X_train_data = X_train_data.reshape((1, X_train_data.shape[0]))

    return X_train_data

'''
get_prediction methods is auto encoder method that gives a lowe dimension repersentation on high dimension data
'''

def get_prediction(X_train, loss_func, SEED=42, LR=0.5, EPOCHS=10):
    tf.random.set_seed(SEED)

    encoder = tf.keras.Sequential([
        tf.keras.layers.Dense(768)
    ])
    decoder = tf.keras.Sequential([
        tf.keras.layers.Dense(X_train.shape[1])
    ])

    autoencoder = tf.keras.Sequential([encoder, decoder])

    optimizer = tf.keras.optimizers.SGD(learning_rate=LR)
    autoencoder.compile(loss= loss_func, optimizer=optimizer)

    history = autoencoder.fit(X_train, X_train, epochs=EPOCHS)
    return encoder.predict(X_train)