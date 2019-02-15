# -*- coding: utf-8 -*-
import wx

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


class MiAplicacion(wx.Frame):
    modelo = ''
    txt1 = ''
    txt2 = ''
    vectorizer = ''
    log = ''
    # FUNCIONES
    # ----------------------------------------------------------------------------

    def addToLog(self, text):
        global txt2
        txt2.write(text+'\n')

    def clean_corpus(self, corpus):
        xcorpus = corpus.get_values()
        for i in range(len(corpus)):
            xcorpus[i] = re.sub("[^a-zA-Z]", " ", corpus[i].lower())
            xcorpus[i] = ' '.join(xcorpus[i].split())
        return xcorpus

    def tokenize(self, text):
        tokens = nltk.word_tokenize(text,  language='spanish')
        #tokens = nltk.word_tokenize(text,  language='english')

        stems = []
        for item in tokens:
            stems.append(nltk.PorterStemmer().stem(item))
        return stems

    def train_test_vector(self, xtrain, xtest):
        global vectorizer
        vectorizer = TfidfVectorizer(
            tokenizer=self.tokenize, max_df=0.80, use_idf=True, min_df=1)
        # vectorizer    = CountVectorizer(min_df=1,binary=True) # metrica binaria
        vector_train = vectorizer.fit_transform(xtrain)
        vector_test = vectorizer.transform(xtest)
        return vector_train, vector_test, vectorizer

    def preparar(self):
        self.addToLog("Prepararando datos...")
        # 1. CARGAR DATOS encuesta con cabecera texto, clase
        data = pd.read_csv(
            'conSWconPOS.csv',
            # sep=';;;#',
            engine='python'
        )

        # 2. CORPUS
        corpus = self.clean_corpus(data['A'])

        # 3. entrenamiento
        xtrain, xtest, ytrain, ytest = tts(corpus, data['B'], test_size=0.30)

        # 4. TOKENIZACION + VECTORIZACION
        global vectorizer
        xtrain, xtest, vectorizer = self.train_test_vector(
            xtrain=xtrain, xtest=xtest)

        # 5. MODELO SVM
        global modelo
        modelo = svm.SVC(kernel='linear')
        modelo.fit(X=xtrain, y=ytrain)

        # 6. PREDICT + METRICAS
        prediccion = modelo.predict(xtest)

        self.addToLog(pd.crosstab(ytest, prediccion, rownames=[
            'REAL'], colnames=['PREDICCION']).to_string())
        self.addToLog(classification_report(ytest, prediccion))

    def procesar(self, texto):
        Xnew = [texto]
        vector_new = vectorizer.transform(Xnew)
        ynew = modelo.predict(vector_new)
        for i in range(len(Xnew)):
            self.addToLog("X=%s, Predicted=%s" % (Xnew[i], ynew[i]))

    def __init__(self, parent, title):
        print("Creando interfaz grafica...")
        wx.Frame.__init__(self, parent=parent, title=title, size=(900, 500))

        # Paneles
        p1 = wx.Panel(self)
        p2 = wx.Panel(self)

        # Sizers
        mainsz = wx.BoxSizer(wx.VERTICAL)
        p1sz = wx.BoxSizer(wx.HORIZONTAL)
        p2sz = wx.BoxSizer(wx.HORIZONTAL)

        # Controles
        st1 = wx.StaticText(p1, -1, "Predicción")
        global txt1, txt2
        txt1 = wx.TextCtrl(p1, -1)
        st2 = wx.StaticText(p2, -1, "Ingresar Texto")
        txt2 = wx.TextCtrl(p2, -1, style=wx.TE_READONLY | wx.TE_MULTILINE)
        bt = wx.Button(self, -1, " Evaluar ")

        # Agregar elementos a los sizers
        p1sz.Add(st1, 1, wx.EXPAND | wx.ALL, 20)
        p1sz.Add(txt1, 4, wx.EXPAND | wx.ALL, 20)
        p2sz.Add(st2, 1, wx.EXPAND | wx.ALL, 20)
        p2sz.Add(txt2, 4, wx.EXPAND | wx.ALL, 20)
        mainsz.Add(p1, 1, wx.EXPAND)
        mainsz.Add(p2, 3, wx.EXPAND)
        mainsz.Add(bt, 0, wx.ALIGN_CENTRE | wx.ALL, 20)
        bt.Bind(wx.EVT_BUTTON, self.OnClicked)

        # Conf. sizers
        p1.SetSizer(p1sz)
        p2.SetSizer(p2sz)
        self.SetSizer(mainsz)

        self.SetBackgroundColour(p1.GetBackgroundColour())

        self.Centre(True)
        self.preparar()
        self.Show()

    def OnClicked(self, event):
        self.procesar(txt1.GetValue())
        txt1.SetValue("")


if __name__ == '__main__':
    app = wx.App()
    frame = MiAplicacion(None, u"PREDICTOR DE AGRESIVIDAD")
    frame.preparar
    app.MainLoop()
