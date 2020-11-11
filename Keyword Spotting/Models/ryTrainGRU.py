## Usar CONV LSTM 2D PARA ESTE MODELO
##Tratar de recrear los modelos de LSTM para poder adaptarlos a los espectrogramas

''' Proyecto seguridad_alarmas
@Author Daniel Vallejo Aldana, Alfredo Arturo Elías Miranda
@Asesor: Dr.Adrian Pastor López Monroy
'''

import numpy as np
import time
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, AveragePooling1D, SeparableConv1D,SeparableConv2D
from tensorflow.keras.layers import AveragePooling1D,BatchNormalization,AveragePooling2D,SpatialDropout2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import regularizers
from tensorflow.keras.utils import plot_model
basePath= './Audios_entrenar/'
fn= 'archivo_de_entrenamiento.npz'


t0= time.time()

z= np.load(basePath+fn)
print('Audio_cargado')

x_train=    z['x_train']
y_train=    z['y_train']
x_val=     z['x_test']
y_val=     z['y_test']
#x_test =z['x_val']
#y_test=z['y_val']

fnModel= 'modelo_final_audios_lstm.hdf5'

print(".... z= np.load({}) will train into {}".format(fn, fnModel))


# In[]
import tensorflow as tf

def ryFeature(x,
           sample_rate= 30000,

           frame_length= 1024,
           frame_step=    128,  # frame_length//2

           num_mel_bins=     128,
           lower_edge_hertz= 20,     # 0
           upper_edge_hertz= 30000/2, # sample_rate/2

           mfcc_dim= 13
           ):

    stfts= tf.signal.stft(x,
                          frame_length, #=  256, #1024,
                          frame_step, #=    128,
                          #fft_length= 1024
                          pad_end=True
                          )

    spectrograms=     tf.abs(stfts)
    log_spectrograms= tf.math.log(spectrograms + 1e-10)

    #Deformación del espectrograma en escala lineal a la escala de mel
    num_spectrogram_bins= stfts.shape[-1]  #.value

    linear_to_mel_weight_matrix= tf.signal.linear_to_mel_weight_matrix(
          num_mel_bins,
          num_spectrogram_bins,
          sample_rate,
          lower_edge_hertz,
          upper_edge_hertz)

    mel_spectrograms= tf.tensordot(
          spectrograms,
          linear_to_mel_weight_matrix, 1)

    mel_spectrograms.set_shape(
          spectrograms.shape[:-1].concatenate(
              linear_to_mel_weight_matrix.shape[-1:]))

    #Calcular un logaritmo bien definido para obtener la log magnitud de los log-mel espectrogramas
    log_mel_spectrograms= tf.math.log(mel_spectrograms + 1e-10)

    # Calculo de los 13 primeros coeficientes de mel
    mfccs= tf.signal.mfccs_from_log_mel_spectrograms(
          log_mel_spectrograms)[..., :mfcc_dim]
    #print(mfccs.shape)

    feature= {'mfcc':               mfccs,
              'log_mel_spectrogram':log_mel_spectrograms,
              'log_spectrogram':    log_spectrograms,
              'spectrogram':        spectrograms}

    return  feature


# In[]

import time

import tensorflow as tf


def get_all_fearure(all_x, batch_size= 1000):
    t0= time.time()

    x= all_x.astype(np.float32)
    i=0
    XL=[]
    while i < x.shape[0]:

        if i+batch_size<=x.shape[0]:
            xx= x[i:i+batch_size]
        else:
            xx= x[i:]

        XX= ryFeature(xx)
        X= XX['log_mel_spectrogram']
        #'log_spectrogram'] #'mfcc'] #'log_mel_spectrogram']

        X= X.numpy().astype(np.float32)

        i  += batch_size
        XL += [X]

    XL= np.concatenate(XL)
    print('XL.shape={}'.format(XL.shape))

    dt= time.time()-t0
    print('tf.signal.stft, dif_tiempo dt= {}'.format(dt))

    return XL
# In[]
print('.... get_all_fearure() .... ')

t0= time.time()

#X_testREAL= get_all_fearure(x_testREAL)
#X_test=     get_all_fearure(x_test)
X_val=      get_all_fearure(x_val)
X_train=    get_all_fearure(x_train)

dt= time.time()- t0
print('... get_all_fearure() ... dt(sec)= {:.3f}'.format(dt))

nTime, nFreq= X_train[0].shape
print(X_train[0].shape)

#nTime, nFreq= (125, 128)

# In[]
def normalize(x, axis= None):
    if axis== None:
        x= (x-x.mean())/x.std()
    else:
        x= (x-x.mean(axis= axis))/x.std(axis= axis)

    return x

# In[]
print('....Función de normalización....')

X_train= X_train.reshape(-1, nTime, nFreq,1,1).astype('float32')
X_val=   X_val.reshape(-1, nTime, nFreq,1,1).astype('float32')
#X_test=  X_test.reshape( -1, nTime, nFreq,1).astype('float32')
#X_testREAL=  X_testREAL.reshape( -1, nTime, nFreq, 1).astype('float32')

#Normalización de los datos
X_train=     normalize(X_train)#, axis=0)  # normalized for the all set, many utterence
X_val=       normalize(X_val)#, axis=0)
#X_test=      normalize(X_test)#, axis=0)
#X_testREAL=  normalize(X_testREAL)#, axis=0)
#'''

# In[]

import tensorflow as tf

tf.keras.backend.clear_session()
# For easy reset of notebook state.

from tensorflow              import keras
from tensorflow.keras        import layers, Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import AveragePooling1D,SpatialDropout1D,GlobalAveragePooling2D, ConvLSTM2D
from tensorflow.keras.optimizers import Nadam, Adadelta

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#opt=Nadam
#print("Se importo el optimizador "+str(opt))


nCategs= len(set(y_train)) #36 #c_train.size #36
print(nCategs)
print(set(y_train))
frames=len(X_train)
x= Input(shape= (nTime, nFreq,1,1))

h= x
#Necesitamos implementar la capa de atención del modelo para que sea completo al del articulo dado
#esta es la parte de las convoluciones
'''h=layers.TimeDistributed(Conv2D(64,(3,3), padding='same'))(h)
h=layers.TimeDistributed(BatchNormalization())(h)
h=layers.TimeDistributed(MaxPooling2D(2,padding='same'))(h)
h=layers.TimeDistributed(Dropout(0.4))(h)
h=layers.TimeDistributed(Conv2D(64,(3,3), padding='same'))(h)
h=layers.TimeDistributed(BatchNormalization())(h)
h=layers.TimeDistributed(MaxPooling2D(2,padding='same'))(h)
h=layers.TimeDistributed(Dropout(0.4))(h)
h=layers.TimeDistributed(Flatten())(h)
h= layers.TimeDistributed(Dense(4096,activation='relu'))(h)
h=layers.Bidirectional(layers.LSTM(64, return_sequences=True))(h)
h=layers.Bidirectional(layers.LSTM(64))(h)
h= Dense(64,activation='relu')(h)
h= Dropout(0.4)(h)
h= Dense(32,activation='relu')(h)
h= Dropout(0.4)(h)
h= Dense(nCategs, activation='softmax')(h)'''

h=layers.TimeDistributed(Conv2D(32,(20,5),strides=(8,2), padding='same'))(h)
#h=layers.TimeDistributed(BatchNormalization())(h)
#h=layers.TimeDistributed(MaxPooling2D(2,padding='same'))(h)
h=layers.TimeDistributed(Dropout(0.4))(h)
#h=layers.TimeDistributed(Conv2D(64,(3,3), padding='same'))(h)
#h=layers.TimeDistributed(BatchNormalization())(h)
#h=layers.TimeDistributed(MaxPooling2D(2,padding='same'))(h)
#h=layers.TimeDistributed(Dropout(0.4))(h)
h=layers.TimeDistributed(Flatten())(h)
#h= layers.TimeDistributed(Dense(4096,activation='relu'))(h)
h=layers.GRU(32, return_sequences=True)(h)
h=layers.GRU(32)(h)
h= Dense(64,activation='relu')(h)
h= Dropout(0.4)(h)
#h= Dense(32,activation='relu')(h)
#h= Dropout(0.4)(h)
h= Dense(nCategs, activation='softmax')(h)

y= h

m= Model(inputs=  x,
         outputs= y)

m.summary()

#plot_model(m,to_file='modelo_audios,png')
m.compile(
        loss= 'sparse_categorical_crossentropy',
        metrics= ['accuracy'],
        optimizer='Adadelta'
        )


es= EarlyStopping(
        monitor=   'val_loss',
        min_delta= 1e-10,
        patience=  10,
        mode=      'min',
        verbose=   1)



mc= ModelCheckpoint(fnModel,
        monitor=    'val_accuracy',
        verbose=    1,
        save_best_only= True,
        mode=      'max')

t0= time.time()

h= m.fit(X_train, y_train,

        batch_size=100, #1000, # 1000
        epochs=    4500, #5000 épocas para el entrenamiento de las redes neuronales

        callbacks=[mc], #[es, mc], #este es el early stopping de la red neuronal por si no se quiere hacer todas las épocas

        #validation_split= 0.1
        validation_data= (X_val, y_val)
        )


print('[INFO] h= m.fit() ... dt(sec)= {}'.format(dt))
predicciones=m.predict(X_val)
predicciones = np.argmax(predicciones, axis=1)
print("Clases: ambiente, apagar, encender, tranquilizate, secunet")
print(confusion_matrix(y_val,predicciones))

print('[INFO] Se concluye el entrenamiento del modelo, fin de la sesión')
