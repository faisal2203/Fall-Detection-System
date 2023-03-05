import cv2
import mediapipe as mp
import numpy as np
import os
from twilio.rest import Client
def sendmsg():
    account_sid = 'AC802d335077a61423009b8111a61a918e'
    auth_token = 'c0ad5ca9dfeeea5b5562081d9411d32b'
    client = Client(account_sid, auth_token)

    message = client.messages.create(
        messaging_service_sid='MG9e6fa1453aaf02784615cf2c54ca2432',
        body='Fall Down',
        to='+919990339112'
    )
    call = client.calls.create(
        url='https://faisal2203.github.io/fall_detection/voice.xml', #Add the hosted XML file
        to='+919990339112',
        from_='+14174532983'
    )

    print(message.sid)
    print(call.sid)

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

angle=0
#function for detecting the angle
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[1] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle

#loop to access videos from folder

for file in os.listdir("DATASET"):
    path=os.path.join("DATASET", file)
# Create a VideoCapture object and read from input file
    cap= cv2.VideoCapture(path)
    flag=0
    print("-")
#to check if video opens
    if (cap.isOpened() == False):
        print("Error opening video  file")
#Converting input to grey
#Segmenting foreground
#Remove noise and shadow
#detecting objects
    object_detector = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=15)

#if video opens
    while (cap.isOpened()):

# Capture frame-by-frame
        ret, frame = cap.read()
        mask = object_detector.apply(frame)
        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#converting to rgb for pose estimation
        try:
            imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(imgRGB)
        except:
            pass

        try:
            landmarks=results.pose_landmarks.landmark

#pose estimation

            shoulder= [landmarks[mpPose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[mpPose.PoseLandmark.RIGHT_SHOULDER.value].y]
            hip = [landmarks[mpPose.PoseLandmark.RIGHT_HIP.value].x,
                    landmarks[mpPose.PoseLandmark.RIGHT_HIP.value].y]
            knee = [landmarks[mpPose.PoseLandmark.RIGHT_KNEE.value].x,
                    landmarks[mpPose.PoseLandmark.RIGHT_KNEE.value].y]

            calculate_angle(shoulder, hip, knee)

            shoulderl = [landmarks[mpPose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[mpPose.PoseLandmark.RIGHT_SHOULDER.value].y]
            hipl = [landmarks[mpPose.PoseLandmark.RIGHT_HIP.value].x,
                   landmarks[mpPose.PoseLandmark.RIGHT_HIP.value].y]
            kneel = [landmarks[mpPose.PoseLandmark.RIGHT_KNEE.value].x,
                    landmarks[mpPose.PoseLandmark.RIGHT_KNEE.value].y]

            calculate_angle(shoulderl, hipl, kneel)

        except:
            pass

#pose connection
        if results.pose_landmarks:
        #mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            for cnt in contours:
# Calculate area and remove small elements
                area = cv2.contourArea(cnt)
                #print("flag",flag)
                if area > 2000:
                    x, y, w, h = cv2.boundingRect(cnt)

                    while flag == 0:
                        if 0 < calculate_angle(shoulderl, hipl, kneel) < 45 or 0< calculate_angle(shoulder, hip, knee) <45 or 270< calculate_angle(shoulder, hip, knee) >45 or 270< calculate_angle(shoulderl, hipl, kneel) >45  or  calculate_angle(shoulderl, hipl, kneel) <90 or  calculate_angle(shoulder, hip, knee)<90:#0< calculate_angle(shoulderl, hipl, kneel) <45 or 0< calculate_angle(shoulder, hip, knee) <45 or 270< calculate_angle(shoulder, hip, knee) >45 or 270< calculate_angle(shoulderl, hipl, kneel) >45  or  calculate_angle(shoulderl, hipl, kneel) <90 or  calculate_angle(shoulder, hip, knee)<90 or  calculate_angle(shoulderl, hipl, kneel) <45  or calculate_angle(shoulder, hip, knee) <45
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                            fall = True
                            flag=1

                        # elif fall==True:
                            print("Sending SMS")
                            sendmsg()

                        else :
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            fall=False

                        #print(fall)
                # if flag==1:
                #     print(fall)

        if ret == True:
#to display the video
            cv2.imshow("Image", frame)



        # Press Q on keyboard to  exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# Break the loop
        else:
            break