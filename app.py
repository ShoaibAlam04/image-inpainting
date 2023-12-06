import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt

DIRECTORY= r"C:\Users\DELL\Downloads\leaf\leaf"
CATAGORIES= ['Strawberry_fresh','Strawberry_scrotch']

data=[]

for categories in CATAGORIES:
    folder=os.path.join(DIRECTORY,categories)
    label=CATAGORIES.index(categories)
    
    for img in os.listdir(folder):
        img=os.path.join(folder,img)
        img_arr=cv2.imread(img)
        img_arr=cv2.resize(img_arr,(100,100))
        
        data.append([img_arr,label])

data


random.shuffle(data)

x=[]
y=[]


for features,label in data:
    x.append(features)
    y.append(label)


x=np.array(x)
y=np.array(y)

x

x=x/255

x

x.shape

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation

model=Sequential()
model.add( Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add( Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(2,activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(x,y,epochs=15,validation_split=0.1)
