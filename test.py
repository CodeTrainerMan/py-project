# encoding:utf-8
import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw

def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "font/simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


thres = 0.45 # Threshold to detect object
nms_threshold = 0.5

cap = cv2.VideoCapture("rtsp://admin:Admin123@172.16.10.226:554")
cap.set(3,1280)
cap.set(4,720)
cap.set(10,70)

classNames= []
classFile = 'coco.names'
with open(classFile,'rt',encoding='utf-8') as f:
    classNames = f.read().rstrip('\n').split('\n')
    print(classNames[0])

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success,img = cap.read()
    classIds, confs, bbox = net.detect(img,confThreshold=thres)
    # print(classIds,bbox)
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1,-1)[0])
    confs = list(map(float,confs))

    indices = cv2.dnn.NMSBoxes(bbox,confs,thres,nms_threshold)
    for i in indices:
        i = i[0]
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        cv2.rectangle(img,(x,y),(x+w,h+y),color=(0,255,0),thickness=2)
        img = cv2ImgAddText(img, classNames[classIds[i][0]-1], box[0] + 80, box[1] + 30, (0, 255, 0), 20)
        cv2.putText(img,classNames[classIds[i][0]-1],(box[0]+10,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    # if len(classIds) != 0:
    #     for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
    #         cv2.rectangle(img,box,color=(0,255,0),thickness=2)
    #         # img = cv2ImgAddText(img,classNames[classId-1],(box[0]+10,box[1]+30),
    #         #             cv2.FONT_HERSHEY_COMPLEX,(0,255,0),2)
    #
    #         img = cv2ImgAddText(img, classNames[classId-1], box[0]+10, box[1]+30, (0,255,0), 20)
    #         # cv2.putText(img,classNames[classId-1],(box[0]+10,box[1]+30),
    #         #             cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    #         img = cv2ImgAddText(img, str(round(confidence*100,2)), box[0] + 80, box[1] + 30, (0, 255, 0), 20)
    #         # cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
    #         #             cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    cv2.imshow("Output",img)
    cv2.waitKey(1)


