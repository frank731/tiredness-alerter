import time

import winrt.windows.ui.notifications as notifications
import winrt.windows.data.xml.dom as dom
import cv2
import os
from keras.models import load_model
import numpy as np
import dlib
from imutils import face_utils
from scipy.spatial import distance

app = '{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\\WindowsPowerShell\\v1.0\\powershell.exe'

nManager = notifications.ToastNotificationManager
notifier = nManager.create_toast_notifier(app)

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
score = 0
yawn_score = 0
can_alert = True

EAR_THRESHOLD = 0.5
MAR_THRESHOLD = 1.5
YAWN_EAR_THRESHOLD = 0.57

cam = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]


def calc_eye(eye):
    return (distance.euclidean(eye[1], eye[5]) + distance.euclidean(eye[2], eye[4])) / (2.0 * distance.euclidean(eye[0], eye[3]))


def calc_mouth(mouth):
    return (distance.euclidean(mouth[2], mouth[10]) + distance.euclidean(mouth[4], mouth[8])) / distance.euclidean(mouth[0], mouth[6])


def alert(title, content):
    global notifier
    tString = """
      <toast>
        <visual>
          <binding template='ToastGeneric'>
            <text>{}</text>
            <text>{}</text>
          </binding>
        </visual>
        <actions>
          <action
            content="Delete"
            arguments="action=delete"/>
          <action
            content="Dismiss"
            arguments="action=dismiss"/>
        </actions>        
      </toast>
    """.format(title, content)

    xDoc = dom.XmlDocument()
    xDoc.load_xml(tString)
    notifier.show(notifications.ToastNotification(xDoc))


while True:
    ret, frame = cam.read()
    if not ret:
        print("No frame found")
        continue
    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray, 0)
    for face in faces[:1]:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouthPoints = shape[mStart:mEnd]
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouthPoints)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouthHull], -1, (0, 0, 255), 1)
        ar = calc_eye(leftEye) + calc_eye(rightEye)
        mar = calc_mouth(mouthPoints)
        if ar < EAR_THRESHOLD:
            score += 1
        else:
            score -= 1
        if mar > MAR_THRESHOLD and ar < YAWN_EAR_THRESHOLD:
            yawn_score += 4
        else:
            yawn_score -= 4
        cv2.putText(frame, "AR" + str(ar), (200, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "MAR" + str(mar), (200, height - 40), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    score = max(0, min(score, 120))
    yawn_score = max(0, min(yawn_score, 110))
    if score + yawn_score > 100 and can_alert:
        cv2.putText(frame, "ALERT", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        can_alert = False
        alert("You look tired", "Go take a break!")
    elif score + yawn_score < 20:
        can_alert = True
    cv2.putText(frame, 'Score:' + str(score + yawn_score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
