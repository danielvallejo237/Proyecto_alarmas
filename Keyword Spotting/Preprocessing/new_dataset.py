from tqdm import tqdm
import os
import librosa
import numpy as np
import time
import sys
import pylab as pl

Dic_cats = {
	'__unknown__':0,
	'ambiente':0, 'apagar':1, 'encender':2, 'tranquilizate':3, 'secunet':4
}

Cats_list = Dic_cats.keys()


def _getFileCategory(file, catDic):
	categ = file.split('/')[-2]
	return catDic.get(categ, 0)


def ryLengthNormalize(x, length=30000):
	''' En esta función consideramos los casos posibles de entrada de un archivo, cuando la entrada es
	más grande, más pequeña o de un tamaño considerable '''
	if len(x)== length:
		X=x
	elif len(x)> length:
	 randPos= np.random.randint(len(x)-length)
	 X=x[randPos:randPos+length]

	else:
		randPos= np.random.randint(length-len(x))
		X=np.random.random(length)*1e-10
		X[randPos:randPos+len(x)]=x
	return X


if __name__ == '__main__':
	basePath = sys.argv[1] #path donde se encuentran todos los datos
 #Obtenemos las ctegorias para entrenamiento y para prueba y lo dividimos en una proporción de 80-20
	#obetenmos los nombres de los archivos a usar.
	train_files = []
	for cat in os.listdir(basePath + 'train/'):
		for file in os.listdir(basePath + 'train/' +cat):
			if file.split('.')[-1].lower() != 'wav':
				continue
			train_files.append(basePath + 'train/' + cat + '/' + file)

	test_files = []
	for cat in os.listdir(basePath + 'test/'):
		for file in os.listdir(basePath + 'test/' +cat):
			if file.split('.')[-1].lower() != 'wav':
				continue
			test_files.append(basePath + 'test/' + cat + '/' + file)

	val_files = []
	for cat in os.listdir(basePath + 'val/'):
		for file in os.listdir(basePath + 'val/' +cat):
			if file.split('.')[-1].lower() != 'wav':
				continue
			val_files.append(basePath + 'val/' + cat + '/' + file)


	train_labels = [_getFileCategory(f, Dic_cats) for f in train_files]
	test_labels = [_getFileCategory(f, Dic_cats) for f in test_files]
	val_labels = [_getFileCategory(f, Dic_cats) for f in val_files]

	trainWAVlabelsDict = dict(zip(train_files, train_labels))
	testWAVlabelsDict = dict(zip(test_files, test_labels))
	valWAVlabelsDict = dict(zip(val_files,val_labels))

	trainInfo = {'files':train_files, 'labels':trainWAVlabelsDict}
	testInfo = {'files':test_files, 'labels':testWAVlabelsDict}
	valInfo= {'files':val_files, 'labels':valWAVlabelsDict}
	gscInfo = {'train':trainInfo,
				'test':testInfo,
				'val':valInfo
				}


	xLL= []
	yLL= []

	for s in ['test', 'train','val']:
	    aL=  gscInfo[s]['files']
	    xL= []
	    #print(s, "Info:", aL)
	    for fn in tqdm(aL):
	        x, sr= librosa.load(fn, sr= None)
	        x= ryLengthNormalize(x)
	        xL += [x]
	    xL= np.vstack(xL)
	    xLL += [xL]
	    yL=  list(gscInfo[s]['labels'].values())
	    yL= np.array(yL)
	    yLL += [yL]

	x_test, x_train,x_val= xLL
	y_test, y_train,y_val= yLL


	assert x_train.shape[0]        == y_train.shape[0]
	assert x_test.shape[0]         == y_test.shape[0]
	assert x_val.shape[0]		   == y_val.shape[0]

	x_train= 	x_train.astype('float32')
	x_test=     x_test.astype('float32')
	x_val= 		x_val.astype('float32')
	y_train= 	y_train.astype('int')
	y_test=     y_test.astype('int')
	y_val= 		y_val.astype('int')

	t0 = time.time()

	print(y_train)
	print(y_test)

	fn= 'archivo_de_entrenamiento.npz'
	if not os.path.isfile(basePath+fn):
	    np.savez_compressed(
	        basePath+fn,
	        x_train=    x_train,
	        y_train=    y_train,
	        x_test=     x_test,
	        y_test=     y_test,
			x_val=  x_val,
			y_val =  y_val,
	        )

	dt= time.time()-t0
	print(f'np.savez_compressed(), fn= {fn}, dt(sec)= {dt:.2f}')
	print("[INFO] Archivo npz guardado")
