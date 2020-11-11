import os
import numpy as np
import sys
import shutil

dataPath=sys.argv[1] # Es el nombre del path donde se encuentran nuestros audios
    #Creación de las carpetas
try:
    basePath=sys.argv[2]
except:
    basePath=os.getcwd() #get current directory
Categorias=os.listdir(dataPath+'/Manuel/')
personas=os.listdir(dataPath)
print('Las categorias encontradas son',Categorias)
for categoria in Categorias:
    if not os.path.exists(basePath +'/Splited_audio/train/'+categoria):
        os.makedirs(basePath +'/Splited_audio/train/'+categoria)
for categoria in Categorias:
    if not os.path.exists(basePath +'/Splited_audio/test/'+categoria):
        os.makedirs(basePath +'/Splited_audio/test/'+categoria)
for categoria in Categorias:
    if not os.path.exists(basePath +'/Splited_audio/val/'+categoria):
        os.makedirs(basePath +'/Splited_audio/val/'+categoria)

for persona in personas:
    for categoria in Categorias:
        files=[file for file in os.listdir(dataPath+'/'+persona+'/'+categoria) if file.split('.')[-1].lower()=='wav']
        np.random.shuffle(files)
        train_FileNames,test_FileNames = np.split(np.array(files),[int(len(files)*0.80)])
        train_FileNames,val_FileNames =np.split(train_FileNames,[int(train_FileNames.shape[0]*0.9)])
        train_FileNames = [dataPath+'/'+persona+'/'+categoria+'/'+ name for name in train_FileNames.tolist()]
        test_FileNames = [dataPath+'/'+persona+'/'+categoria+'/' + name for name in test_FileNames.tolist()]
        val_FileNames=[dataPath+'/'+persona+'/'+categoria+'/' + name for name in val_FileNames.tolist()]
        for name in train_FileNames:
            shutil.copy(name,basePath +'/Splited_audio/train/'+categoria)
        for name in test_FileNames:
            shutil.copy(name,basePath +'/Splited_audio/test/'+categoria)
        for name in val_FileNames:
            shutil.copy(name,basePath +'/Splited_audio/val/'+categoria)
print("[INFO] Audios separados del conjunto de datos inicial")
