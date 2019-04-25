import sys
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation, Dropout
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras import backend as K

#Reiniciando la sesion
K.clear_session()

data_entrenamiento = 'data/entrenamiento'
data_validacion = 'data/validacion'
path = 'data/entrenamiento'
os.path.isdir(path)

#Iniciando las variables de entrada
#Parametros
epocas = 20
altura, longitud = 100, 100
batch_size = 32
pasos = 1000
pasos_validacion = 200
filtros_Conv1 = 32
filtros_Conv2 = 64
tamano_filtro1 = (3, 3)
tamano_filtro2 = (2, 2)
tamano_pool = (2, 2)
clases = 5
lr = 0.00005

#Preprocesamiento de la imagenes de las flores
entrenamiento_datagen = ImageDataGenerator(
    rescale = 1. /255,
    shear_range = 0.3,
    zoom_range = 0.3,
    horizontal_flip = True
)

validacion_datagen = ImageDataGenerator(
    rescale = 1. /255
)

imagen_entrenamiento = entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    target_size = (altura, longitud),
    batch_size = batch_size,
    class_mode = 'categorical'
)

imagen_validacion = validacion_datagen.flow_from_directory(
    data_validacion,
    target_size = (altura, longitud),
    batch_size = batch_size,
    class_mode = 'categorical'
)


#Crear la red CNN
cnn = Sequential()

#Se agrega el primer filtro de convolucion
cnn.add(Convolution2D(
    filtros_Conv1, 
    tamano_filtro1, 
    padding= 'same', 
    input_shape = (altura, longitud, 3), 
    activation = 'relu'))

#Se reliza el max Polling para reducir las imagenes
cnn.add(MaxPooling2D(pool_size = tamano_pool))

#Se vuelve a realizar otro filtro de convolucion
cnn.add(Convolution2D(
    filtros_Conv2,
    tamano_filtro2,
    padding = 'same',
    activation = 'relu'
))
#Se realiza otro max Pooling para reducir la imagen
cnn.add(MaxPooling2D(pool_size = tamano_pool))

#Se mete toda esa informacion en un solo vector
cnn.add(Flatten())

#Se crean 256 neuronas las cuales se activan con la funcion Relu
cnn.add(Dense(256, activation = 'relu'))

#Desecha la mitad de las imagenes para que se sobreajuste la red
cnn.add(Dropout(0.5))

cnn.add(Dense(clases, activation = 'softmax'))

cnn.compile(loss = 'categorical_crossentropy', optimizer=optimizers.Adam(lr=lr), metrics = ['accuracy'])


cnn.fit(imagen_entrenamiento, 
        steps_per_epoch = pasos, 
        epochs = epocas, 
        validation_data = imagen_validacion,
        validation_steps = pasos_validacion
       )

dir = './modelo'

if not os.path.exists(dir):
    os.mkdir(dir)
cnn.save('./modelo/modelo_Fl.h5')
cnn.save_weights('./modelo/pesos_Fl.h5')