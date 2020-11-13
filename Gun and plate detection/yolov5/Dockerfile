# Start FROM Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvcr.io/nvidia/pytorch:20.08-py3

# Install dependencies 
RUN apt-get update
RUN apt-get install 'ffmpeg'\
    'libsm6'\ 
    'libxext6'  -y
RUN pip install --upgrade pip
RUN pip install opencv-python>=4.1.2
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install gsutil

ADD . .


CMD ['python', 'detect.py','--weights','(weights)','--source','(source)']

