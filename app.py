from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

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

camera = cv2.VideoCapture(0)
import time
time.sleep(2)

def faceBox(net, frame):

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300),
                                 [104,117,123], swapRB=False)

    net.setInput(blob)
    detections = net.forward()

    bboxs = []

    for i in range(detections.shape[2]):

        confidence = detections[0,0,i,2]

        if confidence > 0.75:

            x1 = int(detections[0,0,i,3] * w)
            y1 = int(detections[0,0,i,4] * h)
            x2 = int(detections[0,0,i,5] * w)
            y2 = int(detections[0,0,i,6] * h)

            bboxs.append([x1,y1,x2,y2])

    return bboxs


def generate_frames():
    while True:
        success, frame = camera.read()

        if not success:
            continue

        # Face detection
        bboxs = faceBox(faceNet, frame)

        for bbox in bboxs:
            x1, y1, x2, y2 = bbox

            face = frame[y1:y2, x1:x2]

            if face.shape[0] < 50 or face.shape[1] < 50:
                continue

            blob = cv2.dnn.blobFromImage(face, 1.0, (227,227),
                                         MODEL_MEAN_VALUES, swapRB=False)

            genderNet.setInput(blob)
            gender = genderList[genderNet.forward()[0].argmax()]

            ageNet.setInput(blob)
            age = ageList[ageNet.forward()[0].argmax()]

            label = f"{gender}, {age}"

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,label,(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)

        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
   