import cv2
import mediapipe as mp
import os
import numpy as np
import importlib
from anglecalculator import calculate_angle

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose



def pull_up():

    cap = cv2.VideoCapture(0)
    counter = 0
    stage = None

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            results = pose.process(image)
        
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            try:
                landmarks = results.pose_landmarks.landmark
                state = True

                if (landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].visibility < 0.5) and (landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility < 0.5):
                    cv2.putText(image, 'DIRSEKLER ALGILANMADI', (190,22), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                    state = False

                if state:
                
                    shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                    angle = calculate_angle(shoulder, elbow, wrist)

                    cv2.putText(image, str(angle).split('.')[0], 
                                    tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )
                    
                    if angle > 130:
                        stage = "down"
                    if angle < 130 and stage == "down":
                        counter +=1
                        print(counter)
                        stage = "up"

            except:
                pass

            cv2.rectangle(image, (0,0), (160,40), (245,117,16), -1)
            
            # Rep data
            cv2.putText(image, 'PULL UP', (10,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE', (100,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (40,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )               
            image = cv2.resize(image, (1000, 800))
            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def push_up():
    
    cap = cv2.VideoCapture(0)
    counter = 0
    stage = None

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            results = pose.process(image)
        
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            try:
                state = True
                landmarks = results.pose_landmarks.landmark


                if (landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].visibility < 0.5) and (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].visibility < 0.5):
                    cv2.putText(image, 'AYAKLAR ALGILANMADI', (190,22), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                    state = False
                    
                    
                if (landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility < 0.5) and (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility < 0.5):
                    cv2.putText(image, 'BACAK ALGILANMADI', (190,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                    state = False

                if state:
                    shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    
                    angle = calculate_angle(shoulder, elbow, wrist)

                    bacak_angle = calculate_angle(hip, knee, ankle)

                    cv2.putText(image, str(angle).split('.')[0], 
                                    tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )
                    
                    cv2.putText(image, str(bacak_angle).split('.')[0], 
                                    tuple(np.multiply(knee, [640, 480]).astype(int)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )
                    
                    if bacak_angle > 150:
                        if angle > 100:
                            stage = "up"
                        if angle < 100 and stage == "up":
                            counter +=1
                            print(counter)
                            stage = "down"   
                    else:
                        cv2.putText(image, 'BACAK ACISINI DOGRU YAPINIZ', (190,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

            except:
                pass

            
                

            cv2.rectangle(image, (0,0), (160,40), (245,117,16), -1)
            
            # Rep data
            cv2.putText(image, 'PUSH UP', (10,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE', (100,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (40,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )               
            image = cv2.resize(image, (1000, 800))
            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()



def barbell_biceps_curl():
    print("barbell_biceps_curl fonksiyonu çağrıldı")

    cap = cv2.VideoCapture(0)
    counter = 0
    stage = None

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            results = pose.process(image)
        
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            try:
                state = True
                landmarks = results.pose_landmarks.landmark


                if (landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].visibility < 0.5) and (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].visibility < 0.5):
                    cv2.putText(image, 'AYAKLAR ALGILANMADI', (190,27), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                    state = False

                if (landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].visibility < 0.5) and (landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility < 0.5):
                    cv2.putText(image, 'DIRSEKLER ALGILANMADI', (190,42), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                    state = False
                    
                if (landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility < 0.5) and (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility < 0.5):
                    cv2.putText(image, 'BACAK ALGILANMADI', (190,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                    state = False

                if state:
                    shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    
                    angle = calculate_angle(shoulder, elbow, wrist)

                    bacak_angle = calculate_angle(hip, knee, ankle)

                    cv2.putText(image, str(angle).split('.')[0], 
                                    tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )
                    
                    cv2.putText(image, str(bacak_angle).split('.')[0], 
                                    tuple(np.multiply(knee, [640, 480]).astype(int)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )
                    
                    if bacak_angle > 160:
                        if angle > 100:
                            stage = "down"
                        if angle < 100 and stage == "down":
                            counter +=1
                            print(counter)
                            stage = "up"   
                    else:
                        cv2.putText(image, 'BACAK ACISINI DOGRU YAPINIZ.', (190,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

            except:
                pass

            
                

            cv2.rectangle(image, (0,0), (160,40), (245,117,16), -1)
            
            # Rep data
            cv2.putText(image, 'Barbell Biceps Curl', (10,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        
            cv2.putText(image, stage, 
                        (40,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )               
            image = cv2.resize(image, (1000, 800))
            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def bench_press():
    print("bench_press fonksiyonu çağrıldı")

def chest_fly_machine():
    print("chest_fly_machine fonksiyonu çağrıldı")

def deadlift():
    print("deadlift fonksiyonu çağrıldı")

def decline_bench_press():
    print("decline_bench_press fonksiyonu çağrıldı")

def hammer_curl():
    print("hammer_curl fonksiyonu çağrıldı")

def hip_thrust():
    print("hip_thrust fonksiyonu çağrıldı")

def incline_bench_press():
    print("incline_bench_press fonksiyonu çağrıldı")

def lat_pulldown():
    print("lat_pulldown fonksiyonu çağrıldı")

def lateral_raises():
    print("lateral_raises fonksiyonu çağrıldı")

def leg_extension():
    print("leg_extension fonksiyonu çağrıldı")

def leg_raises():
    print("leg_raises fonksiyonu çağrıldı")

def plank():
    print("plank fonksiyonu çağrıldı")

def romanian_deadlift():
    print("romanian_deadlift fonksiyonu çağrıldı")

def russian_twist():
    print("russian_twist fonksiyonu çağrıldı")

def shoulder_press():
    print("shoulder_press fonksiyonu çağrıldı")

def squat():
    print("squat fonksiyonu çağrıldı")

def t_bar_row():
    print("t_bar_row fonksiyonu çağrıldı")

def tricep_dips():
    print("tricep_dips fonksiyonu çağrıldı")

def tricep_pushdown():
    print("tricep_pushdown fonksiyonu çağrıldı")
