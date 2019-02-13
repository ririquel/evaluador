# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 13:45:39 2018

@author: Christian
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
import nltk
import nltk.data
# nltk.download('punkt')
from sklearn.model_selection import train_test_split as tts
import re
#from sklearn.neural_network import MLPClassifier


# FUNCIONES
# ----------------------------------------------------------------------------
def clean_corpus(corpus):
    xcorpus = corpus.get_values()
    for i in range(len(corpus)):
        xcorpus[i] = re.sub("[^a-zA-Z]", " ", corpus[i].lower())
        xcorpus[i] = ' '.join(xcorpus[i].split())
    return xcorpus


def tokenize(text):
    tokens = nltk.word_tokenize(text,  language='spanish')
    #tokens = nltk.word_tokenize(text,  language='english')

    stems = []
    for item in tokens:
        stems.append(nltk.PorterStemmer().stem(item))
    return stems


def train_test_vector(xtrain, xtest):
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize, max_df=0.80, use_idf=True, min_df=1)
    # vectorizer    = CountVectorizer(min_df=1,binary=True) # metrica binaria
    vector_train = vectorizer.fit_transform(xtrain)
    vector_test = vectorizer.transform(xtest)
    return vector_train, vector_test, vectorizer


# EJECUCION
# ----------------------------------------------------------------------------
# 1. CARGAR DATOS encuesta con cabecera texto, clase
data = pd.read_csv('corpus/csvEncuesta.csv')

# 2. CORPUS
corpus = clean_corpus(data['texto'])

# entrenamiento
xtrain, xtest, ytrain, ytest = tts(corpus, data['clase'], train_size=0.70)

# 4. TOKENIZACION + VECTORIZACION
xtrain, xtest, vectorizer = train_test_vector(xtrain=xtrain, xtest=xtest)

# 5. MODELO SVM
modelo = svm.SVC(kernel='linear')
modelo.fit(X=xtrain, y=ytrain)

# 6. PREDICT + METRICAS
prediccion = modelo.predict(xtest)

print(pd.crosstab(ytest, prediccion, rownames=[
      'REAL'], colnames=['PREDICCION']))
print(classification_report(ytest, prediccion))
Xnew = [
    'la clase es mala',
    'excelente'
]
vector_new = vectorizer.transform(Xnew)
ynew = modelo.predict(vector_new)
for i in range(len(Xnew)):
    print("X=%s, Predicted=%s" % (Xnew[i], ynew[i]))
