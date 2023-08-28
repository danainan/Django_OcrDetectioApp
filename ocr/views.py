from django.shortcuts import render
import cv2
from django.http import HttpResponse, StreamingHttpResponse
from django.shortcuts import render
from django.views import View
import threading
from PIL import Image
from tesserocr import PyTessBaseAPI , RIL, OEM, PSM
import os 
from django.conf import settings
import pytesseract
import numpy as np
import time
import imutils
from imutils.object_detection import non_max_suppression
import argparse


                

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = self.frame
        # _, jpeg = cv2.imencode('.jpg', image)
        # return jpeg.tobytes()
        return self.frame

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()

def boxes(frame):
    ocr_path = os.path.join(settings.OCR_ROOT, 'tessdata_best-main')
    frame_array = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    with PyTessBaseAPI(path=ocr_path, lang='tha+eng') as api:
        api.SetImage(frame_array)
        api.set_page_seg_mode(PSM.AUTO)
        api.set_oem(OEM.LSTM_ONLY)
        boxes = api.GetComponentImages(RIL.TEXTLINE, True)
        for i, (im, box, _, _) in enumerate(boxes):
            cv2.rectangle(frame, (box['x'], box['y']), (box['x'] + box['w'], box['y'] + box['h']), (0, 255, 0), 2)
    _, jpeg = cv2.imencode('.jpg', frame)


    return jpeg.tobytes()
    

def gen(camera):
    # while True:
    #     frame = camera.get_frame()

        
        
    #     yield(b'--frame\r\n'
    #           b'Content-Type: image/jpeg\r\n\r\n' + boxes(frame) + b'\r\n\r\n')
    net = cv2.dnn.readNet(os.path.join(settings.OCR_ROOT, 'frozen_east_text_detection.pb'))
    while True: 
            frame = camera.get_frame()
            frame = imutils.resize(frame, width=1000)
                      
            (inpWidth, inpHeight) = (320, 320)
            rW = frame.shape[1] / float(inpWidth)
            rH = frame.shape[0] / float(inpHeight)
            blob = cv2.dnn.blobFromImage(frame, 1.0, (inpWidth, inpHeight), (123.68, 116.78, 103.94), True, False)
            net.setInput(blob)
            (scores, geometry) = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])
            (numRows, numCols) = scores.shape[2:4]
            rects = []
            confidences = []
            for y in range(0, numRows):
                scoresData = scores[0, 0, y]
                xData0 = geometry[0, 0, y]
                xData1 = geometry[0, 1, y]
                xData2 = geometry[0, 2, y]
                xData3 = geometry[0, 3, y]
                anglesData = geometry[0, 4, y]
                for x in range(0, numCols):
                    if scoresData[x] < 0.5:
                        continue
                    (offsetX, offsetY) = (x * 4.0, y * 4.0)
                    angle = anglesData[x]
                    cos = np.cos(angle)
                    sin = np.sin(angle)
                    h = xData0[x] + xData2[x]
                    w = xData1[x] + xData3[x]
                    endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                    endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                    startX = int(endX - w)
                    startY = int(endY - h)
                    rects.append((startX, startY, endX, endY))
                    confidences.append(scoresData[x])
            boxes = non_max_suppression(np.array(rects), probs=confidences)
            for (startX, startY, endX, endY) in boxes:
                startX = int(startX * rW)
                startY = int(startY * rH)
                endX = int(endX * rW)
                endY = int(endY * rH)
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                
            
       


           
            
            

            



            frame = cv2.imencode('.jpg', frame)[1].tobytes()
            yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            

def camera_feed(request):
   try:
        cam = VideoCamera()
        return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
   except:  # This is bad!
        pass
    
        

def index(request):
    return render(request, 'index.html')

