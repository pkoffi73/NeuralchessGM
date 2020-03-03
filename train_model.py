# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 23:57:00 2019

@author: phili
"""


import numpy as np
import tensorflow as tf

'''CONSTANTS:
- START: File number to start with when resuming training
- NUM_EPOCHS: Number of epochs for training
- FILE_NUMBERS: Numbers of data files
- RESNET_NUMBER: Number of resnet blocks of the NN
- BATCH_SIZE, LEARNING RATE: NN training parameters
- FEATURES: Features number of the chess board represenation
- change_learning_rate: if true, a new learning rate is put in the optimizer
- load_model: if true, an existing model is loaded for further training
- name_directory_data: name of the directory including the training dataset
- name_model: Neural Network model name
'''

FILE_NUMBERS = 21
FEATURES = 22
RESNET_NUMBER = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.0001
START = 0
NUM_EPOCHS = 20
change_learning_rate = False
name_directory_data = "processedGM6/"
name_model = 'model/saved_model_value_policy.h5'
load_model = False

resinput = [0 for i in range(RESNET_NUMBER)]

inputs = tf.keras.Input(shape=(8, 8, FEATURES,), dtype='float32')
x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', use_bias=False, name='layer1')(inputs)
x = tf.keras.layers.BatchNormalization(name='layer2')(x)
resinput[0] = tf.keras.activations.relu(x)
#Layer of Resnets
for i in range(RESNET_NUMBER):
    resinput[i] = tf.keras.activations.relu(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', use_bias=False, name='layer' + str(i) + '1')(resinput[i])
    x = tf.keras.layers.BatchNormalization(name='layer' + str(i) + '2')(x)
    x = tf.keras.activations.relu(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', use_bias=False, name='layer' + str(i) + '3')(x)
    x = tf.keras.layers.BatchNormalization(name='layer' + str(i) + '4')(x)
    x = tf.keras.layers.Add()([x, resinput[i]])
#Value Head
y = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), padding='SAME', use_bias=False, name='layer1000')(x)
y = tf.keras.layers.BatchNormalization(name='layer1001')(y)
y = tf.keras.activations.relu(y)
y = tf.keras.layers.Flatten()(y)
y = tf.keras.layers.Dense(256, name='layer1002', use_bias=False)(y)
y = tf.keras.layers.BatchNormalization(name='layer1003')(y)
y = tf.keras.activations.relu(y)
y = tf.keras.layers.Dense(1, name='layer1004', use_bias=False)(y)
y = tf.keras.layers.BatchNormalization(name='layer1005')(y)
outputs1 = tf.keras.activations.tanh(y)
#Policy Head
z = tf.keras.layers.Conv2D(filters=2, kernel_size=(1, 1), padding='SAME', use_bias=False, name='layer2000')(x)
z = tf.keras.layers.BatchNormalization(name='layer2001')(z)
z = tf.keras.activations.relu(z)
z = tf.keras.layers.Flatten()(z)
z = tf.keras.layers.Dense(1917, name='layer2002', use_bias=False)(z)
z = tf.keras.layers.BatchNormalization(name='layer2003')(z)
outputs2 = tf.keras.activations.softmax(z)



if __name__ == '__main__':    
    #model initialization
    if not load_model:
        model = tf.keras.Model(inputs=inputs, outputs=[outputs1, outputs2])
        print('model built')
        opti = tf.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=0.9)
        losses = ['mse', 'categorical_crossentropy']
        #weights between value loss and policy loss
        lossWeights = [0.5, 1.0]
        model.compile(optimizer=opti, loss=losses, loss_weights=lossWeights, metrics=['accuracy'])
        print('model compiled')
        callbacks=[tf.keras.callbacks.TensorBoard(),\
                   tf.keras.callbacks.ModelCheckpoint(filepath='model/model_weights_best_bitmap_CNN.hdf5', \
                                                      verbose=1, save_best_only=False, \
                                                      save_weights_only=True, \
                                                      monitor='val_accuracy',\
                                                         mode='max')]
    #model loading if resume training... and optionnally changing learning rate
    else:
        callbacks=[tf.keras.callbacks.TensorBoard(),\
                   tf.keras.callbacks.ModelCheckpoint(filepath='model/model_weights_best_bitmap_CNN.hdf5', \
                                                      verbose=1, save_best_only=False, \
                                                      save_weights_only=True, \
                                                      monitor='val_accuracy',\
                                                      mode='max')]
        
        model = tf.keras.models.load_model(name_model)
        if change_learning_rate:
            opti = tf.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=0.9)
            losses = ['mse', 'categorical_crossentropy']
            lossWeights = [0.5, 1.0]
            model.compile(optimizer=opti, loss=losses, loss_weights=lossWeights, metrics=['accuracy'])
            

    for i in range(NUM_EPOCHS):
        print("Epoch #%s / %s" % (i, NUM_EPOCHS))
        a = list(np.arange(FILE_NUMBERS)) * 2
        files = a[START:START + FILE_NUMBERS]
        for j in files:
            name = 'processedGM6/dataset_3_classes_bitmap' + str(j) +'.npz'
            with np.load(name) as dat:
                X = dat['arr_0']
                Y1 = dat['arr_1']
                Y2 = dat['arr_2']
            #Validation data
            X_val = X[-10000:]
            Y1_val = Y1[-10000:]
            Y2_val = Y2[-10000:]
            #Training data
            X = X[:-10000]
            Y1 = Y1[:-10000]
            Y2 = Y2[:-10000]
            model.fit(X, [Y1, Y2], validation_data = (X_val, [Y1_val, Y2_val]), batch_size=128, shuffle=True, epochs=1, verbose=1, callbacks=callbacks)
            #at the end of each file training session, keras session is cleared
            #to prevent memory issues
            #So model needs to be saved and reloaded
            X, Y = 0, 0
            model.save(name_model)
            print('model saved')
            tf.keras.backend.clear_session()
            model = tf.keras.models.load_model(name_model)
            print('model loaded')
