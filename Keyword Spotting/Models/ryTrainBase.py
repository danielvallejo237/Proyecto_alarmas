''' Proyecto seguridad_alarmas
Entrenamiento del modelo base para comparación
'''

import numpy as np
import time
import os
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, AveragePooling1D, SeparableConv1D,SeparableConv2D
from tensorflow.keras.layers import AveragePooling1D,BatchNormalization,AveragePooling2D,SpatialDropout2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import regularizers
from tensorflow.keras.utils import plot_model
import tensorflow as tf
from datetime import datetime
from packaging import version
basePath= './Splited_audio/'
fn= 'archivo_de_entrenamiento.npz'
import tensorboard

try:
    os.rmdir("./logs")
except:
    pass

if not os.path.exists("./logs"):
    os.makedirs("./logs")

t0= time.time()

z= np.load(basePath+fn)
print('Audio_cargado')

x_train=    z['x_train']
y_train=    z['y_train']
x_val=     z['x_val']
y_val=     z['y_val']
x_test =z['x_test']
y_test=z['y_test']

fnModel= 'modelo_final_audios_basemodel.hdf5'

print(".... z= np.load({}) will train into {}".format(fn, fnModel))

# Define the Keras TensorBoard callback.
logdir="./logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)



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
X_test=     get_all_fearure(x_test)
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

X_train= X_train.reshape(-1, nTime, nFreq,1).astype('float32')
X_val=   X_val.reshape(-1, nTime, nFreq,1).astype('float32')
X_test=  X_test.reshape( -1, nTime, nFreq,1).astype('float32')
#X_testREAL=  X_testREAL.reshape( -1, nTime, nFreq, 1).astype('float32')

#Normalización de los datos
X_train=     normalize(X_train)#, axis=0)  # normalized for the all set, many utterence
X_val=       normalize(X_val)#, axis=0)
X_test=      normalize(X_test)#, axis=0)
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
from tensorflow.keras.layers import AveragePooling1D,SpatialDropout1D,GlobalAveragePooling2D
from tensorflow.keras.optimizers import Nadam

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#opt=Nadam
#print("Se importo el optimizador "+str(opt))


nCategs= len(set(y_train)) #36 #c_train.size #36
print(nCategs)
print(set(y_train))


x= Input(shape= (nTime, nFreq,1))

h= x


#esta es la parte de las convoluciones
h= Conv2D(8,   (16,16), activation='relu', padding='same')(h)
h= MaxPooling2D((4,4), padding='same')(h)
h= Dropout(0.2)(h)

h= Conv2D(16,   (8,8), activation='relu', padding='same')(h)
h= MaxPooling2D((4,4), padding='same')(h)
h= Dropout(0.2)(h)

h= Flatten()(h)

h= Dense(256,  activation='relu')(h)
h= Dropout(0.2)(h)


h= Dense(nCategs,  activation='softmax')(h)

y= h

m= Model(inputs=  x,
         outputs= y)

m.summary()

#plot_model(m,to_file='modelo_audios,png')
m.compile(
        loss= 'sparse_categorical_crossentropy',
        metrics= ['accuracy'],
        #optimizer='Nadam'
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
h= m.fit(X_train, y_train,

        batch_size=100, #1000, # 1000
        epochs=    2500, #5000 épocas para el entrenamiento de las redes neuronales

        callbacks=[mc,tensorboard_callback], #[es, mc], #este es el early stopping de la red neuronal por si no se quiere hacer todas las épocas

        #validation_split= 0.1
        validation_data= (X_val, y_val)
        )


print('[INFO] h= m.fit() ... dt(sec)= {}'.format(dt))
results = m.evaluate(X_test, y_test, batch_size=100)
print("Pérdida del conjunto de prueba, Acc del conjunto de prueba:", results)
predicciones=m.predict(X_test)
predicciones =np.argmax(predicciones, axis=1)
print("Clases: ambiente, apagar, encender, tranquilizate, secunet")
print(confusion_matrix(y_test,predicciones))
print("Accuracy score",accuracy_score(y_test,predicciones))
print("Macro f1 score",f1_score(y_test,predicciones, average='macro'))
print("Micro f1 score",f1_score(y_test,predicciones, average='micro'))
print("Recall",recall_score(y_test,predicciones,average='macro'))
print("Precision",precision_score(y_test,predicciones,average='macro'))
print("[INFO] Imprimiendo medidas y resultados")

print('[INFO] Se concluye el entrenamiento del modelo, fin de la sesión')
