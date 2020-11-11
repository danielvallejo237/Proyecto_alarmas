
''' Reconocimiento en tiempo real de voz para el modelo de secunet'''
import time

import numpy as np

import tensorflow as tf

from tensorflow.keras.models import load_model
import argparse
import sounddevice as sd
from tensorflow.keras.layers import Input
#import winsound
#import beepy

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
    #print(spectrograms.shape)
    log_spectrograms= tf.math.log(spectrograms + 1e-10)

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

    # Calcular la log magnitud de los espectrogramas .
    log_mel_spectrograms= tf.math.log(mel_spectrograms + 1e-10)
    #Calcular MFCCs de los log mel espectrogramas y tomar los primeros trece coeficientes
    mfccs= tf.signal.mfccs_from_log_mel_spectrograms(
          log_mel_spectrograms)[..., :mfcc_dim]
    #print(mfccs.shape)

    feature= {'mfcc':               mfccs,
              'log_mel_spectrogram':log_mel_spectrograms,
              'log_spectrogram':    log_spectrograms,
              'spectrogram':        spectrograms}

    return  feature

def normalize(x):
    x= (x-x.mean())/x.std()
    return x


from datetime import datetime

def predict(x, withProb= False):#, fs=16000):
    global NNmodel, LabelDic
    #print(x.shape)
    prob=  NNmodel.predict(x)#.reshape(1,fs,1)
    index= np.argmax(prob[0])
    y= LabelDic[index]

    if withProb== True:
        probability= np.max(prob[0])
        y= (y, probability)
    return y

def recWav(x, featureOut= False, withProb= False):
    x= x.flatten()
    X= ryFeature(x)['log_mel_spectrogram']
    #print(X.shape)

    X= X.numpy().astype(np.float32)

    X= normalize(X)
    #print(X.shape)
    Xin= X.reshape(1,X.shape[0],X.shape[1], 1)
    y=   predict(Xin, withProb)

    if featureOut == True:
        return y, X
    else:
        return y
import time

def rec_long_wav(x= None, T=1, dt=1, fs=30000, pauseByKey= False, fn= None,mtime=True):

    if pauseByKey==True:
        aKey= input('Presionar una letra para grabar el sonido...')
    try:
        if fn==None and x.all() == None:
                x= sd.rec(int(T*fs),
                    samplerate= fs,
                    channels=   1,
                    dtype=      'float32')
                sd.wait()  # Esperar hasta que se
        elif fn != None:
            x= np.load(fn)
        else:
            print('x.shape= {}'.format(x.shape))
            pass
    except:
        if fn==None and x == None:
            x= sd.rec(int(T*fs),
                samplerate= fs,
                channels=   1,
                dtype=      'float32')
            sd.wait()  # Esperar hasta que se

        elif fn != None:
            x= np.load(fn)
        else:
            print('x.shape= {}'.format(x.shape))
            pass
    T= x.size/fs
    if T==1:
        y= recWav(x)
    elif T>1:
        t=0
        yL= []
        while t<T-dt:
            #En esta parte es donde se debe de encontrar el overlapp
            if int((1+t)*fs)<=T*fs: #Partimos los frames sin overlapp en partes de la frecuencia deseada
                x1sec= x[int(t*fs) : int(t*fs)+fs]
            else:
                x1sec= np.random.random(1*fs)*1e-10 #Residuos de los audios
                x1sec= x1sec.astype(np.float32)
                xx= x[int(t*fs): ].flatten()
                x1sec[0:xx.size]= xx
            y= recWav(x1sec, withProb= True)
            yL += [y]
            t += dt
        y= np.array(yL)
    else:
        y= None
        pass
    state=0
    secret_key=0
    comand=list()
    skey=list()
    cont=0
    conts=0
    #print(y)
    for i in range(len(y)):
        cont+=1
        conts+=1
        if cont>10 and state==1:
            state=0
            cont=0
        if conts>10 and secret_key==1:
            secret_key=0
            conts=0
        if y[i][0] in ['secunet'] and float(y[i][1])>=0.99 and state is not 1:
            state+=1
            comand.append(y[i][0])
        if state==1 and y[i][0] in ['apagar', 'encender'] and float(y[i][1])>=0.90:
            state+=1
            comand.append(y[i][0]) #Esto nos dirá el comando pronunciado por la persona
        if y[i][0] in ['tranquilizate'] and float(y[i][1])>=0.85:
            secret_key+=1
            skey.append(y[i][0])
        if state==2: #Esto nos dice si hay orden o no
            print('Comando {} encontrado a las {}'.format(" ".join(comand),datetime.now()))
            del comand[:]
            if mtime==True:
                print('Tiempo transcurrido de audio es {}'.format(time.time()))
            state=0 #Se regresa el estado al valor inicial
        #En esta parte se puede mejorar haciendo que solo espere un determinado numero de frames y no todo el audio
        if secret_key==2:
            print('Clave secreta {} activada a las {}'.format(" ".join(skey),datetime.now()))
            del skey[:]
            if mtime==True:
                print('Tiempo transcurrido de audio es {}'.format(time.time()))
            secret_key=0 #Las regresamos al valor inicial para poder escuchar una nuea orden
        #Lo anterior nos cubre los casos posibles de todas las alarmas para su activación
    return x, y
'''Podemos notar que el modelo falla en el comando encender ya que no lo detecta correctamente después de decir
La palabra secunet, se debe de aumentar el conjunto de datos para poder obtener un mejor rendimiento, la palabra
tranquilizate es larga lo cual dificulta la correcta idenrtificación de esta palabra, puede cambiarse el traslape de los frames para
poder obtener mejores resultados.
'''

LabelDic= ryGscList=['ambiente','apagar', 'encender', 'tranquilizate','secunet']

#son 5 las categorías que se van a probar
def logarithm_activation(x):
    return tf.math.log(abs(x)+1)



tf.keras.backend.clear_session()

fnModel= './models/pre_trained_models/modelo_final_audios_proposed_model.hdf5'

NNmodel=load_model(fnModel,custom_objects = {"logarithm_activation":logarithm_activation})

import soundfile as sf

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--audio',help="Tenemos el audio de entrada o grabamos el audio",default=None)
    parser.add_argument('--time_duration',help='Duracion del audio de entrada', default=10)
    args= parser.parse_args()
    if args.audio!=None:
        print('[INFO] Cargando audio')
        try:
            data,f= sf.read(args.audio,dtype='float32')
            long=data.shape[0]
            #print(data)
            try:
                if data.shape[1]>1:
                    sgn=[round((data[:,0][i]+data[:,0][1])/max(1e-100,abs(data[:,0][i]+data[:,0][1]))) for i in range(long)]
                    #print(sgn)
                    data=[0.5*sgn[i]*min(0.999999,abs(data[:,0][i]+data[:,0][1])) for i in range (long)]
                    data=np.asarray(data,dtype='float32')
            except:
                pass
            print(data.shape)
        except:
            raise NameError('No se encuentra el archivo de audio {}'.format(args.audio))
        x, y= rec_long_wav(x=data)
    elif args.audio==None:
        print('[INFO] Grabando audio')
        timeDuration= int(args.time_duration) #se
        input('Presiona una tecla para grabar un audio de {} segundos...'.format(
                timeDuration))
        x, y= rec_long_wav(T=timeDuration)
    # .... main recognition ...., en esta parte se deben de reconocer los frames que se pongan en el conjunto de prueba
        xyL= []
        while True:
            aKey= input('Preaionar "q" para salir, o cualquier otra para grabar un nuevo audio {} sec wav...'.format(
                timeDuration))
            if aKey == 'q': break
            x, y= rec_long_wav(T=timeDuration) #Análisis de un wav de larga duración, en donde se detiene si hay un comando
            #para la activación o desactivació de la alarma con el comando secunet
            xyL += (x, y)

    print('Sesion terminada')
