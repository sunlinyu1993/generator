import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Convolution2D,AveragePooling2D
import keras

def generate_arrays_from_file(path):
    while True:
        f = open(path)
        batch_size=64
        count=0
        x=[]
        y=[]
        for line in f:
            xs = line.split(',')
            xs=list(map(lambda x:float(x),xs))
            x_temp=np.array(xs[0:-1])
            y_temp=int(xs[-1])
            count+=1
            x.append(x_temp)
            y.append(y_temp)
            if count==batch_size:
                x=np.concatenate(x,axis=0)
                x=np.reshape(x,(batch_size,60,40,1),order='F')
                y=np.array(y)
                y=keras.utils.to_categorical(y, 5)
                yield x,y
                x = []
                y = []
                count=0
        f.close()
model=Sequential()
model.add(Convolution2D(filters=20,kernel_size=5,strides=1,input_shape=(60,40,1)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=2,strides=2))
model.add(Convolution2D(filters=50,kernel_size=5,strides=1))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=2,strides=2))
model.add(Flatten())
model.add(Dense(600))
model.add(Dense(5))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
model.fit_generator(
    generate_arrays_from_file('F:/DataSet/newmat.csv'),
    steps_per_epoch=129,
    epochs=50)
