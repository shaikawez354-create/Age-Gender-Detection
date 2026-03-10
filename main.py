import cv2
import os

faceProto = "deploy.prototxt"
faceModel = "res10_300x300_ssd_iter_140000.caffemodel"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

ageList = ['(0-2)','(4-6)','(8-12)','(15-20)','(25-32)','(38-43)','(48-53)','(60-100)']
genderList = ['Male','Female']

padding = 20   # ⭐ improves accuracy


def faceBox(net, frame):

    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300),
                                 [104,117,123], swapRB=False)

    net.setInput(blob)
    detections = net.forward()

    bboxs = []

    for i in range(detections.shape[2]):

        confidence = detections[0,0,i,2]

        if confidence > 0.75:   # higher threshold = better quality

            x1 = int(detections[0,0,i,3] * w)
            y1 = int(detections[0,0,i,4] * h)
            x2 = int(detections[0,0,i,5] * w)
            y2 = int(detections[0,0,i,6] * h)

            bboxs.append([x1,y1,x2,y2])

    return bboxs


video = cv2.VideoCapture(0)

# optional HD camera
video.set(3,1280)
video.set(4,720)

while True:

    ret, frame = video.read()
    if not ret:
        break

    bboxs = faceBox(faceNet, frame)

    for bbox in bboxs:

        x1 = max(0, bbox[0] - padding)
        y1 = max(0, bbox[1] - padding)
        x2 = min(frame.shape[1], bbox[2] + padding)
        y2 = min(frame.shape[0], bbox[3] + padding)

        face = frame[y1:y2, x1:x2]

        # skip very small faces
        if face.shape[0] < 50 or face.shape[1] < 50:
            continue

        blob = cv2.dnn.blobFromImage(face, 1.0, (227,227),
                                     MODEL_MEAN_VALUES, swapRB=False)

        genderNet.setInput(blob)
        gender = genderList[genderNet.forward()[0].argmax()]

        ageNet.setInput(blob)
        age = ageList[ageNet.forward()[0].argmax()]

        label = f"{gender}, {age}"

        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    cv2.imshow("Age Gender Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()