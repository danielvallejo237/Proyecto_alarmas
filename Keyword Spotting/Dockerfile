FROM tensorflow/tensorflow
ADD real_time_recognizer.py ./
ADD modelo_final_audios_new.hdf5 ./
ADD requirements.txt ./
RUN apt-get update && apt-get install libportaudio2 libportaudiocpp0 portaudio19-dev libasound-dev libsndfile1-dev -y && pip3 install pyaudio
RUN pip3 install -r requirements.txt
RUN pip3 install SoundFile
ADD prueba_apagar.wav ./
CMD [ "python","./real_time_recognizer.py",'--audio','(audio)']
