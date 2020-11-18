# Proyecto_alarmas

  
 # Agradecimientos
 
 Los autores de este proyecto agradecen el apoyo a la Secretaría de Innovación, Ciencia y Educación Superior del Estado de Guanajuato (SICES) brindado a través del proyecto número SICES/CONV/435/2019

###############################################################################################################################################################
## Keyword spotting 

Different machine learning models to voice command recognition over a certain data set. To obtain the data set used in this porject please contact {pastor.lopez,daniel.vallejo,alfredo.elias}@cimat.mx 

To run the Keyword Spotting part of this repository, follow the commands below

#### pip3 install -r requirements.txt

#### python real_time_recognizer.py

With arguments

#### time_duration 
This determines the duration in seconds of the audio to record

#### audio
An audio file to read and then apply commands recognition


## Handgun detection
To make the handgun detection, we used YOLOV5. To train this model we used images from VOC and other guns and plates images. To get the dataset, please contact the authors. Refer to https://github.com/ultralytics/yolov5 for the full documentation of how to use YOLOV5.

To run predict do:

#### cd "Gun and plate detection"/yolov5

#### python detect.py --weights best.pt

To predict in real time from webcam add <b> --source 1 </b> to the command. If you want to make predictions from a folder, then add <b> --source <directory> </b> and the predictions will be in the <b>inference/outputs </b> directory.
  
  You can use the Docker included to runt the algorithms. 
