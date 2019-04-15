# -*- coding: utf-8 -*-
import wx
# from tkinter import *
# from tkinter.ttk import *
# from tkinter import filedialog
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import BernoulliNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
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
    Nuevas_entradas = ''
    metrica1=""
    metrica2=""
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
        self.addToLog("----------COMENTARIOS EVALUADOS----------")
        # 1. CARGAR DATOS encuesta con cabecera texto, clase
        data = pd.read_csv(
            'sinSWsinPOS.csv',
            # sep=';;;#',
            engine='python'
        )

        # 2. CORPUS
        corpus = self.clean_corpus(data['A'])

        # 3. entrenamiento
        xtrain, xtest, ytrain, ytest = tts(corpus, data['B'], test_size=0.30)
        # xtrain, xtest, ytrain, ytest = tts(corpus, data['B'], train_size=1469)

        # 4. TOKENIZACION + VECTORIZACION
        global vectorizer
        xtrain, xtest, vectorizer = self.train_test_vector(
            xtrain=xtrain, xtest=xtest)

        # 5. MODELO SVM
        global modelo
        # modelo=BernoulliNB()
        # modelo=tree.DecisionTreeClassifier(criterion='entropy',max_depth=29)
        # modelo=RandomForestClassifier(n_estimators=10)
        modelo = svm.SVC(kernel='linear')
        modelo.fit(X=xtrain, y=ytrain)
        # modelo.fit(X=xtrain.toarray(), y=ytrain)
        # print(modelo.tree_.max_depth)

        # 6. PREDICT + METRICAS
        prediccion = modelo.predict(xtest)

        # precisionSVM = cross_val_score(estimator=modelo,X=xtrain, y=ytrain,cv=10)
        # print('Precision SVM LINEAL: {}'.format(precisionSVM))
        # print('Precision SVM LINEAL promedio: {0: .3f} +/- {1: .3f}'.format(np.mean(precisionSVM),np.std(precisionSVM)))
        # self.addToLog("")
        # self.addToLog('Precision SVM LINEAL: {}'.format(precisionSVM))
        # self.addToLog('Precision SVM LINEAL promedio: {0: .3f} +/- {1: .3f}'.format(np.mean(precisionSVM),np.std(precisionSVM)))
        # self.addToLog("")
        # self.addToLog(pd.crosstab(ytest, prediccion, rownames=[
        #     'REAL'], colnames=['PREDICCION']).to_string())
        # self.addToLog("")
        # self.addToLog(classification_report(ytest, prediccion))
        global metrica1
        global metrica2
        metrica1 = pd.crosstab(ytest, prediccion, rownames=['REAL'], colnames=['PREDICCION']).to_string()
        metrica2 = classification_report(ytest, prediccion)

    def procesar(self, texto):
        Xnew = [texto]
        vector_new = vectorizer.transform(Xnew)
        ynew = modelo.predict(vector_new)
        for i in range(len(Xnew)):
            self.addToLog("%s, %s" % (Xnew[i], ynew[i]))



    def __init__(self, parent, title):
        print("Creando interfaz grafica...")
        wx.Frame.__init__(self, parent=parent, title=title, size=(800, 600))

        # Paneles
        p1 = wx.Panel(self)
        p2 = wx.Panel(self)
        p3 = wx.Panel(self)


        # Sizers
        mainsz = wx.BoxSizer(wx.VERTICAL)
        p1sz = wx.BoxSizer(wx.HORIZONTAL)
        p2sz = wx.BoxSizer(wx.HORIZONTAL)
        p3sz = wx.BoxSizer(wx.HORIZONTAL)


        # Controles
        st1 = wx.StaticText(p1, -1, "Ingresar Comentario")
        global txt1, txt2
        txt1 = wx.TextCtrl(p1, -1, style=wx.TE_MULTILINE) 
        st2 = wx.StaticText(p2, -1, "Métricas y Predicción")
        txt2 = wx.TextCtrl(p2, -1, style=wx.TE_READONLY | wx.TE_MULTILINE)
        bt = wx.Button(p3, wx.ID_ANY, " Evaluar ", pos=(100,10))
        bt1 = wx.Button(p3, wx.ID_ANY, " Cargar Archivo ")
        bt2 = wx.Button(p3, wx.ID_ANY, " Guardar Evaluación ")
        bt3 = wx.Button(p3, wx.ID_ANY, " Guardar Métrica y Evaluación ")

        # Agregar elementos a los sizers
        p1sz.Add(st1, 1, wx.EXPAND | wx.ALL, 20)
        p1sz.Add(txt1, 4, wx.EXPAND | wx.ALL, 20)
        p2sz.Add(st2, 1, wx.EXPAND | wx.ALL, 20)
        p2sz.Add(txt2, 4, wx.EXPAND | wx.ALL, 20)
        p3sz.Add(bt, -1, wx.ALIGN_BOTTOM | wx.ALL, 20)
        p3sz.Add(bt1, -1, wx.ALIGN_BOTTOM | wx.ALL, 20)
        p3sz.Add(bt2, -1, wx.ALIGN_BOTTOM | wx.ALL, 20)
        p3sz.Add(bt3, -1, wx.ALIGN_BOTTOM | wx.ALL, 20)
        mainsz.Add(p1, 2, wx.EXPAND)
        mainsz.Add(p2, 4, wx.EXPAND)
        mainsz.Add(p3, 2, wx.EXPAND)

        bt.Bind(wx.EVT_BUTTON, self.OnClicked)
        bt1.Bind(wx.EVT_BUTTON, self.LecturaTxt)
        bt2.Bind(wx.EVT_BUTTON, self.EscrituraTxt)
        bt3.Bind(wx.EVT_BUTTON, self.EscrituraTxt2)
       

        # Conf. sizers
        p1.SetSizer(p1sz)
        p2.SetSizer(p2sz)
        p3.SetSizer(p3sz)
        

        imagen = wx.StaticBitmap(p3, -1, wx.Bitmap('LogoSomos.jpeg', wx.BITMAP_TYPE_ANY), pos = wx.Point (400,0), size = (347,80))
        imagen = wx.StaticBitmap(p3, -1, wx.Bitmap('LogoUbb.jpeg', wx.BITMAP_TYPE_ANY), pos = wx.Point (30,0), size = (347,80))



        self.SetSizer(mainsz)

        self.SetBackgroundColour(p1.GetBackgroundColour())

        self.Centre()
        self.Show()
        self.preparar()

    def OnClicked(self, event):
        self.procesar(txt1.GetValue())
        txt1.SetValue("")


    def LecturaTxt(self, event):
        filename = wx.FileDialog(frame, "Open", "", "", "Archivo de texto|*.txt", wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        filename.ShowModal()
        archivo = open(filename.GetPath(), "r")
        lectura=archivo.read()
        print(lectura)
        Nuevas_entradas=lectura.split(sep='\n')
        i=0
        while i< len(Nuevas_entradas):
           self.procesar(Nuevas_entradas[i])
           i=i+1

    def EscrituraTxt(self, event):
        filename = wx.FileDialog(self, "Guardar Archivo", wildcard="Archivo de Texto (.txt)|.txt",
                       style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
      
        filename.ShowModal()
        if filename:
            f = open(filename.GetPath(), 'a')
            SALIDA = txt2.GetValue()
            SALIDAAUX = ''.join(SALIDA)
            f.write(SALIDAAUX)
            f.close()
    
    def EscrituraTxt2(self, event):
        filename = wx.FileDialog(self, "Guardar Archivo", wildcard="Archivo de Texto (.txt)|.txt",
                       style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
      
        filename.ShowModal()
        if filename:
            f = open(filename.GetPath(), 'a')
            SALIDA = metrica1+'\n\n'+metrica2+'\n\n'+txt2.GetValue()
            
            SALIDAAUX = ''.join(SALIDA)
            f.write(SALIDAAUX)
            f.close()


# Inicializa pantalla principal
if __name__ == '__main__':
    app = wx.App()
    frame = MiAplicacion(None, u"PREDICTOR DE AGRESIVIDAD")
    frame.preparar
    app.MainLoop()
