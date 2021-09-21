#!/usr/bin/env python

# -*- coding: UTF-8 -*-

'''
@Author ：gaoan

@Date ：2021/9/3 21:38
'''
import cv2 as cv
import time
import face_recognition
import numpy as np
import os

# 1.以 1/4 分辨率处理每个视频帧（尽管仍以全分辨率显示）
# 2.仅检测其他视频帧中的人脸。

video_capture = cv.VideoCapture(0)

# 创建已知面部编码及其名称的阵列
known_face_encodings = []
known_face_names = []

basepath = 'known_face' #存放已知面部信息的文件夹
for fram in os.listdir(basepath):
    #获取人物名称，（文件名）
    fram_name = fram[0:-4]
    known_face_names.append(fram_name)
    #识别人脸
    face_image = face_recognition.load_image_file(fram)
    #生成人脸代码
    face_encoding = face_recognition.face_encodings(face_image)[0]
    known_face_encodings.append(face_encoding)


# 初始化某些变量
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes


faceProto = "face_models/opencv_face_detector.pbtxt"
faceModel = "face_models/opencv_face_detector_uint8.pb"

ageProto = "face_models/age_deploy.prototxt"
ageModel = "face_models/age_net.caffemodel"

genderProto = "face_models/gender_deploy.prototxt"
genderModel = "face_models/gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Load network
ageNet = cv.dnn.readNet(ageModel, ageProto)
genderNet = cv.dnn.readNet(genderModel, genderProto)
faceNet = cv.dnn.readNet(faceModel, faceProto)

# 打开视频文件、图像文件或摄像机流
cap = cv.VideoCapture(0)
padding = 20
while cv.waitKey(1) < 0:
    # Read frame
    t = time.time()
    hasFrame, frame = cap.read()

    # 将视频帧调整为 1/4 尺寸，以便更快地进行人脸识别处理
    small_frame = cv.resize(frame, (0, 0), fx=0.25, fy=0.25)
    #将图像从 BGR 颜色（OpenCV 使用）转换为 RGB 颜色（face_recognition使用）
    rgb_small_frame = small_frame[:, :, ::-1]
    # 仅处理其他视频帧以节省时间
    if process_this_frame:

        # 查找当前视频帧中的所有面部和面部编码
        face_locations = face_recognition.face_locations(rgb_small_frame)  #视频中的面部编码
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # 查看该脸是否与已知面部匹配
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding,tolerance=0.65)
            name = "Unknown"

            # 如果在known_face_encodings中发现了匹配项，只需使用第一个匹配项。
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            #或者使用已知面孔与新面孔距离最近的
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # 显示结果
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        #标注面部
        # cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        #在面部下方，写名称
        cv.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv.FILLED)
        font = cv.FONT_HERSHEY_DUPLEX
        cv.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # cv.imshow('Video', frame)


    if not hasFrame:
        cv.waitKey()
        break

    frameFace, bboxes = getFaceBox(faceNet, frame)
    if not bboxes:
        print("No face Detected, Checking next frame")
        continue

    for bbox in bboxes:
        # print(bbox)
        face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]

        blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        # print("Gender Output : {}".format(genderPreds))
        print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        print("Age Output : {}".format(agePreds))
        print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))

        label = "{},{}".format(gender, age)
        cv.putText(frameFace, label, (bbox[0], bbox[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
        cv.imshow("Age Gender Demo", frameFace)
    print("time : {:.3f} ms".format(time.time() - t))


cap.release()
cv.destroyAllWindows()