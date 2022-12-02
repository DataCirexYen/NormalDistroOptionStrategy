##TODAS LAS CALLS Y LAS PUTS ESTAN AL REVES

from pandas_datareader import data
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.express as px
import numpy as np
import math
import yfinance as yf
import pandas as pd
import timeit
import scipy.stats as stats
import seaborn as sns
def input_float(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print('That is not a valid number.')


ticker ='GOOG'


dataDF = yf.download(ticker,'2020-1-1')['Adj Close']
CurrentPrice=yf.Ticker(ticker).info['regularMarketPrice']

dataDF=dataDF.pct_change().iloc[2:].reset_index()
dataDF = dataDF.drop('Date', axis=1)
dataDF=dataDF.groupby(dataDF.index).sum(1)

def popularidad(fila):
    resultado=fila["Adj Close"]+1 #Indexo a cual columna de la fila quiero acceder 
    return (resultado)

dataDF["Adj Close"]=dataDF.apply(popularidad, axis=1)

N = 2
dataDF=dataDF.groupby(dataDF.index // N).prod()
dataDFL=len(dataDF)
print(dataDF.describe())

#fig = px.bar(dataDF, x=dataDF.index, y='Adj Close')
#fig.update_layout(yaxis_range=[0.8,1.15])
#fig.show()
#---
#ax = dataDF.plot.kde()
#plt.show()
#grouped[grouped["followers"]>2000]

def reales(fila):
    resultado=fila["Adj Close"]* CurrentPrice
    return (resultado)

dataDF["Adj Close"]=dataDF.apply(reales, axis=1)


fg = sns.displot(dataDF['Adj Close'], stat="percent")
labels = []
labelsNumbers = []

for ax in fg.axes.ravel():
    
    # add annotations
    for c in ax.containers:

        # custom label calculates percent and add an empty string so 0 value bars don't have a number
        #0.1 means 1 number after comma
           
        for v in c:
            h = v.get_height()
            if h > 0:
                label = f'{h:0.1f}%'
            else:
                label = ""
            labels.append(label)

        for v in c:
            h = v.get_height()
            g =  (v.get_x() + v.get_width() / 2)
            if h > 0:
                labelNumbers = f'{g:0.1f}'
            else:
                labelNumbers = ""
            labelsNumbers.append(labelNumbers)


        ax.bar_label(c, labels=labelsNumbers, label_type='edge', fontsize=8, rotation=90,padding=10)
        ax.bar_label(c, labels=labels, label_type='center', fontsize=8, rotation=0)

ax.margins(y=0.2)
Blank=""

for word in list(labels):  # iterating on a copy since removing will mess things up
    if word in Blank:
        labels.remove(word)
        labelsNumbers.remove(word)

d = {'percentage':labels,'Price':labelsNumbers}
df = pd.DataFrame(d)
print(df)

plt.show()


def p2f(x):
     return float(x["percentage"].strip('%'))/100

df["percentage"]=df.apply(p2f, axis=1)



def callsF():

    res = next(x for x, val in enumerate(df["percentage"])
                                    if val > 0.01)
    global CallNumPerc
    global CallNumPer
    CallNumPerc=df["percentage"].iloc[res]
    CallNumPer=float(df.loc[df['percentage'].eq(CallNumPerc),'Price'].tolist()[0])
    def price2f(x):
        return float(x["Price"])
    
    df["Price"]=df.apply(price2f, axis=1)


    calls=df[(df["percentage"]<0.01) & (df["Price"]<CallNumPer)]

    def LossCalls(x):
        Result=(x["Price"]-CallNumPer)*x["percentage"]
        return Result


    calls["LossCalls"]=calls.apply(LossCalls, axis=1)

    return calls
    #obtener el primer valor mayor 1 
    # Obtener el ultimo valor mayor a 1



def putsF():
    puts=df.iloc[::-1]
    res = next(x for x, val in enumerate(puts["percentage"])
                                    if val > 0.01)
    global PutNumPerc
    global PutNumPer
    PutNumPerc=puts["percentage"].iloc[res] 

    PutNumPer=float(puts.loc[puts['percentage'].eq(PutNumPerc),'Price'].tolist()[0])
    print( {PutNumPer})
    def price2f(x):
        return float(x["Price"])

    puts["Price"]=puts.apply(price2f, axis=1)


    puts=puts[(puts["percentage"]<0.01) & (puts["Price"]>PutNumPer)]

    def LossPuts(x):
        Result=(-x["Price"]+PutNumPer)*x["percentage"]
        return Result


    puts["LossPuts"]=puts.apply(LossPuts, axis=1)

    return(puts)

puts=putsF()
calls=callsF()
#print(calls) #CALLS Y PUTS ESTAN TODAS AL REVES
#print(puts)
print("----------------------------")
print(f'Loss risk de Puts {calls["percentage"].sum()}')
print(f'Loss risk de Calls {puts["percentage"].sum()}')
LossRisk=puts["percentage"].sum()+calls["percentage"].sum()
print(f'Loss risk total {LossRisk}')
print("----------------------------")
print(f'Loss  de Puts {calls["LossCalls"].sum()}')
print(f'Loss  de Calls {puts["LossPuts"].sum()}')
TotalLoss=puts["LossPuts"].sum()+calls["LossCalls"].sum()
print(f'Loss  total {TotalLoss}')
print("----------------------------")
print(f"Poner la Put a este precio {CallNumPer}")
print(f"Poner la Call a este precio {PutNumPer}")

def ratioF():
    CallPrime=input_float(f"$ Prima de la CALL")
    PutPrime=input_float(f"$ Prima de la PUT")
    TotalPrime=(CallPrime+PutPrime)*(1-LossRisk)
    Ratio=TotalPrime/(TotalLoss*-1)
    return(Ratio)

print(ratioF())


def Backtest():
  start_date = '2015-03-09'
  end_date = '2018-02-26'
  goog_data = data.DataReader("GOOG", "yahoo", start_date, end_date)

  goog_data_signal = pd.DataFrame(index=goog_data.index)  
  goog_data_signal["price"] = goog_data["Adj Close"]
  goog_data_signal=goog_data_signal.reset_index()

  NumberOfOperations=int(len(goog_data_signal)/7)
  fallos=0
  aciertos=0
  x=[]
  MoneyLost=[]
  PutMultiplier=CallNumPer/CurrentPrice #Estos estan correcto
  CallMiltiplier=PutNumPer/CurrentPrice   #Estos estan correcto

  for i in range(NumberOfOperations-1):
      b=i+2
      Venta=goog_data_signal["price"].iloc[i]
      call=Venta*CallMiltiplier
      put=Venta*PutMultiplier
      Vence=goog_data_signal["price"].iloc[b]
      if call>Vence and put<Vence:
        x.append(1)
        aciertos+=1
      elif call<Vence:
        x.append(0.0)
        fallos+=1
        MoneyLost.append(Vence-call)
      elif call<Vence:
        x.append(0.0)
        fallos+=1
        MoneyLost.append(put-Vence)
  #if call>Vence and put<Vence:
  #        print("Ganaste")
  Results = pd.DataFrame(x, columns =['Results'])


 
  print(Results)
  fig = plt.figure()
  # Acatiramos como va a hacer el nombre del ax Y. 111 es para el tamaño
  ax1 = fig.add_subplot(111, ylabel="Winrate")

  # Aca hago el grafico de la data de google, btw lw es grosor
  Results["Results"].plot(ax=ax1, color="r", lw=2)

  ax1.text(0, 0.1, f"Numero de fallos:{fallos}", style='italic')
  ax1.text(0, 0.2, f"Numero de aciertos:{aciertos}", style='italic')
  ax1.text(0, 0, f"Ratio de aciertos/fallos:{int(aciertos/fallos)}", style='italic')
  print(f"Perdiste esta guita x cada 1 opcion{MoneyLost}")
  

  plt.show()

Backtest()