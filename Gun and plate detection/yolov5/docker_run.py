import torch
from PIL import Image

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).fuse().eval()  # yolov5s.pt
model = model.autoshape()  # for autoshaping of PIL/cv2/np inputs and NMS

# Images
img1 = Image.open('inference/images/zidane.jpg')
img2 = Image.open('inference/images/bus.jpg')
imgs = [img1, img2]  # batched list of images

# Inference
prediction = model(imgs, size=640)  # includes NMS

print(prediction)

