from django.http import HttpResponse
from django.shortcuts import render
import operator

#manik
import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

def home(request):
    return render(request, 'home.html') # second method I learned

# make home page submit button load count.html
def count(request):
    fulltext = request.GET['text']
    city=request.GET['city']
    date=request.GET['day']
    # if city=="Chittagong" :
    #     url = "https://www.weather-forecast.com/locations/Chittagong/forecasts/latest"
    # else :
    url = "https://www.weather-forecast.com/locations/Chittagong/forecasts/latest"
    d=int(date)*3
    html = urlopen(url)
    soup = BeautifulSoup(html, 'lxml')
    buyers = soup.find_all('span', attrs={'class' : "rain b-forecast__table-value"},limit =d)
    bb=str(buyers)
    b=BeautifulSoup(bb, 'lxml').get_text()
    print("Scraped data")
    print(b)
    su=0
    for i in range(d*3):
        x=b[i]
        if(x!=',' and x!='[' and x!=']' and x!=' ' and x!='-'):
            su = su + int(x)
    print("Historical raindata")
    print(su)
    su1=0
    d1=d-1
    for i in range(d1*3):
        x=b[i]
        if(x!=',' and x!='[' and x!=']' and x!=' ' and x!='-'):
            su1 = su1 + int(x)
    rain=su-su1
    print("rainfall")
    print(rain)

    dataset=pd.read_csv('E:\CSV01.csv')
    Y=dataset.iloc[:,1:7].values
    X=dataset.iloc[:,0].values

    L =  X.tolist()
    Ly = Y.tolist()

    i=L.index(fulltext)
    print("oooooook")
    latlng=Y[i:i+1]

    latlng[0,4]=rain
    latlng[0,5]=su

    lat=latlng[0,0]
    lon=latlng[0,1]
    print(latlng)
    height=latlng[0,2]
    slope=latlng[0,3]
    y_test=[[height,slope,rain,su]]
    #y_test=[[0,0,0,0]]

    # dataset=pd.read_csv('E:\dataset.csv')
    # X=dataset.iloc[:,0:4].values
    # Y=dataset.iloc[:,4].values

    # from sklearn.model_selection  import train_test_split
    # X_train, X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

    # from sklearn.preprocessing import StandardScaler
    # sc=StandardScaler()
    # t=sc.fit_transform(y_test)
    # test1=sc.fit_transform(y_test)
    #y_test=[[76,18,214,190]]
    # print(test1)
    # # load json and create model
    # json_file = open('E:\model1.json', 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # loaded_model = model_from_json(loaded_model_json)
    # # load weights into new model
    # loaded_model.load_weights("E:\model1.h5")
    # print("Loaded model from disk")
    #
    # # evaluate loaded model on test data
    # loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    #
    # y_pred=loaded_model.predict(test1)
    #
    # print(y_pred)
    # #print(y_pred)
    # # result=np.asarray(y_pred)
    # prediction=0
    # prediction=result[0,0]
    # prediction=prediction*100
    dataset=pd.read_csv('E:\dataset.csv')
    X=dataset.iloc[:,0:4].values
    Y=dataset.iloc[:,4].values
    #print(X)
    from sklearn.model_selection  import train_test_split
    X_train, X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

    from sklearn.preprocessing import StandardScaler
    sc=StandardScaler()
    X_train=sc.fit_transform(X_train)
    X_test=sc.transform(X_test)
    y_test=sc.transform(y_test)
    # print(X_train)
    # print(y_test)
    classifier =Sequential()
    #Adding input and first hidden layer
    classifier.add(Dense(output_dim=16,init='uniform',activation='relu',input_dim=4))
    #Adding second hidden layer
    classifier.add(Dense(output_dim=32,init='uniform',activation='relu'))
    #Adding output layer
    classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
    #compiling ANN
    classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    #fitting training set
    classifier.fit(X_train,Y_train,batch_size=9,nb_epoch=30)
    #predicting test set
    y_pred=classifier.predict(y_test)


    result=np.asarray(y_pred)
    #
    prediction=result[0,0]
    prediction=prediction*100
    #Ly[L.index(fulltext)]
    #prediction=0
    # print(index_of_maximum)

    return render(request, 'result.html', {'prediction': prediction,'lat':lat,'lon':lon}) # .items() makes it a list for web page
