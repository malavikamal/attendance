import os
import face_recognition as fr
import cv2
import numpy as np


res = 'invalid'
knownEncodings = []
nameList = []

presentList = []


def check():
    img_names = os.listdir('static\\Pics')
    print(f"images = {img_names}")

    for i in img_names:
        imgTest = fr.load_image_file(os.path.join('static\\Pics', i))
        # imgTest=cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)
        encodeImgTest = fr.face_encodings(imgTest)[0]
        knownEncodings.append(encodeImgTest)

        split = os.path.splitext(i)  # to remove only the extension
        nameList.append(split[0])
    print(f" Names = {nameList}")


def cam():
    global res, presentList, nameList
    res = 'invalid'
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        resizeImg = cv2.resize(img, (0, 0), None, .25, .25)
        resizeImg = cv2.cvtColor(resizeImg, cv2.COLOR_BGR2RGB)

        curFaces = fr.face_locations(resizeImg)
        encodeCurFaces = fr.face_encodings(resizeImg, curFaces)

        for faceEncode, faceLoc in zip(encodeCurFaces, curFaces):
            matches = fr.compare_faces(knownEncodings, faceEncode)
            faceDist = fr.face_distance(knownEncodings, faceEncode)
            print(faceDist)

            matchIndex = np.argmin(faceDist)
            name = 'unknown'

            if matches[matchIndex] and faceDist[matchIndex] < 0.5:
                name = nameList[matchIndex]
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2+25), (x2, y2), (0, 255, 0), -1)
                cv2.putText(img, name, (x1 + 6, y2+20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)
                if name not in presentList:
                    presentList.append(name)
                else:
                    pass
            else:
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 + 25), (x2, y2), (0, 255, 0), -1)
                cv2.putText(img, name, (x1 + 6, y2 + 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)
            print(name)
        #  need to adjust font

        cv2.imshow('Say Cheese :)', img)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

check()
cam()
