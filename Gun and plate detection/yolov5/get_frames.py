import cv2
for i in range(5):
   vidcap = cv2.VideoCapture(f'videos_v2/video{i+1}.mp4')
   success,image = vidcap.read()
   frame = 0
   count = 1
   success = True
   while success:
      success,image = vidcap.read()
      try:
         if frame % 10 == 0:
            cv2.imwrite(f"videos_v2/frames_video{i+1}/images/frame{count}.jpg", image)
            count += 1
      except:
         success=False
      frame += 1
