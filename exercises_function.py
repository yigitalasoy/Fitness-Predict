import cv2
import mediapipe as mp
import os
import numpy as np
import importlib
from anglecalculator import calculate_angle
import random
from datetime import datetime

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def generateRandomExerciseCount():
    randomSayi = random.randint(3,5)
    return randomSayi



def pull_up():
    # 1.3 sn
    randomGoal = generateRandomExerciseCount()


    defaultExerciseTime = 1.3

    startTime = datetime.now()
    finishTime = datetime.now()
    goalTime = float(randomGoal) * defaultExerciseTime
    finishState = True

    print(f"random goal: {randomGoal}")
    print(f"goal time: {goalTime}")


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
                        if counter==1:
                            startTime = datetime.now()
                            print(f"{startTime}")
                        stage = "up"

            except:
                pass

            cv2.rectangle(image, (0,0), (180,80), (245,117,16), -1)

            
            cv2.putText(image, 'PULL UP', (10,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            
            cv2.putText(image, 'STAGE', (100,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (40,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            
            if int(randomGoal) <= int(counter):
                
                cv2.putText(image, "Goal Completed. You can continue if you want",
                        (185,42), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                
                
            
            if (int(randomGoal) == int(counter)) and finishState:
                finishState = False
                finishTime = datetime.now()
                exerciseTime = finishTime - startTime
                print(f"yapılan süre: {exerciseTime.total_seconds()}")

                percent = round(((exerciseTime.total_seconds())*100) / (defaultExerciseTime * randomGoal))
                
                print(f"percent: {percent}")
                if percent > 100:
                    print("100 den büyük. yavaş yapıldı")

                    percent = percent - 100
                    pointLowerCount = percent // 10
                    print(f"pointLowerCount: {pointLowerCount}")
                    if pointLowerCount > 9:
                        point = 1
                    else:
                        point = 10 - pointLowerCount

                    print(f"Point: {point}")

                else:
                    print("100 den küçük. hızlı yapıldı")

                    percent = 100 - percent
                    pointLowerCount = percent // 10
                    print(f"pointLowerCount: {pointLowerCount}")
                    if pointLowerCount > 9:
                        point = 1
                    else:
                        point = 10 - pointLowerCount

                    print(f"Point: {point}")
                    

                

                
            cv2.putText(image, str(datetime.now()),
                        (185,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

            cv2.putText(image, "Goal: "+str(randomGoal), 
                        (10,52), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            
            
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
    defaultExerciseTime = 1.8

    randomGoal = generateRandomExerciseCount()
    print(f"random goal: {randomGoal}")
    
    startTime = datetime.now()
    finishTime = datetime.now()
    goalTime = float(randomGoal) * defaultExerciseTime
    finishState = True

    print(f"random goal: {randomGoal}")
    print(f"goal time: {goalTime}")

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
                            if counter==1:
                                startTime = datetime.now()
                                print(f"{startTime}")
                            print(counter)
                            stage = "down"   
                    else:
                        cv2.putText(image, 'BACAK ACISINI DOGRU YAPINIZ', (190,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

            except:
                pass

            
                

            cv2.rectangle(image, (0,0), (180,80), (245,117,16), -1)
            
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
            
            if int(randomGoal) <= int(counter):
                
                cv2.putText(image, "Goal Completed. You can continue if you want",
                        (185,42), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                
                
            
            if (int(randomGoal) == int(counter)) and finishState:
                finishState = False
                finishTime = datetime.now()
                exerciseTime = finishTime - startTime
                print(f"yapılan süre: {exerciseTime.total_seconds()}")

                percent = round(((exerciseTime.total_seconds())*100) / (defaultExerciseTime * randomGoal))
                
                print(f"percent: {percent}")
                if percent > 100:
                    print("100 den büyük. yavaş yapıldı")

                    percent = percent - 100
                    pointLowerCount = percent // 10
                    print(f"pointLowerCount: {pointLowerCount}")
                    if pointLowerCount > 9:
                        point = 1
                    else:
                        point = 10 - pointLowerCount

                    print(f"Point: {point}")

                else:
                    print("100 den küçük. hızlı yapıldı")

                    percent = 100 - percent
                    pointLowerCount = percent // 10
                    print(f"pointLowerCount: {pointLowerCount}")
                    if pointLowerCount > 9:
                        point = 1
                    else:
                        point = 10 - pointLowerCount

                    print(f"Point: {point}")
                    

                

                
            cv2.putText(image, str(datetime.now()),
                        (185,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

            cv2.putText(image, "Goal: "+str(randomGoal), 
                        (10,52), 
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
    defaultExerciseTime = 3.0

    randomGoal = generateRandomExerciseCount()
    print(f"random goal: {randomGoal}")

    startTime = datetime.now()
    finishTime = datetime.now()
    goalTime = float(randomGoal) * defaultExerciseTime
    finishState = True

    print(f"random goal: {randomGoal}")
    print(f"goal time: {goalTime}")


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
                            if counter==1:
                                startTime = datetime.now()
                                print(f"{startTime}")
                            print(counter)
                            stage = "up"   
                    else:
                        cv2.putText(image, 'BACAK ACISINI DOGRU YAPINIZ.', (190,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

            except:
                pass

            cv2.rectangle(image, (0,0), (180,80), (245,117,16), -1)
            
            cv2.putText(image, 'Barbell Biceps Curl', (10,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        
            cv2.putText(image, stage, 
                        (40,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            
            if int(randomGoal) <= int(counter):
                
                cv2.putText(image, "Goal Completed. You can continue if you want",
                        (185,42), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                
                
            
            if (int(randomGoal) == int(counter)) and finishState:
                finishState = False
                finishTime = datetime.now()
                exerciseTime = finishTime - startTime
                print(f"yapılan süre: {exerciseTime.total_seconds()}")

                percent = round(((exerciseTime.total_seconds())*100) / (defaultExerciseTime * randomGoal))
                
                print(f"percent: {percent}")
                if percent > 100:
                    print("100 den büyük. yavaş yapıldı")

                    percent = percent - 100
                    pointLowerCount = percent // 10
                    print(f"pointLowerCount: {pointLowerCount}")
                    if pointLowerCount > 9:
                        point = 1
                    else:
                        point = 10 - pointLowerCount

                    print(f"Point: {point}")

                else:
                    print("100 den küçük. hızlı yapıldı")

                    percent = 100 - percent
                    pointLowerCount = percent // 10
                    print(f"pointLowerCount: {pointLowerCount}")
                    if pointLowerCount > 9:
                        point = 1
                    else:
                        point = 10 - pointLowerCount

                    print(f"Point: {point}")
                    

                

                
            cv2.putText(image, str(datetime.now()),
                        (185,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

            cv2.putText(image, "Goal: "+str(randomGoal), 
                        (10,52), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            
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

    defaultExerciseTime = 3.0

    randomGoal = generateRandomExerciseCount()
    print(f"random goal: {randomGoal}")

    

    startTime = datetime.now()
    finishTime = datetime.now()
    goalTime = float(randomGoal) * defaultExerciseTime
    finishState = True

    print(f"random goal: {randomGoal}")
    print(f"goal time: {goalTime}")


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
                    
                if (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility < 0.5) and (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility < 0.5):
                    cv2.putText(image, 'OMUZLAR ALGILANMADI', (190,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                    state = False

                if state:
                    if landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility > landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility:
                        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                        elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                        wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                        #xValue = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x
                        #yValue = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
                        #print("right daha net")
                    else:
                        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                        #xValue = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x
                        #yValue = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
                        #print("left daha net")

                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    
                    angle = calculate_angle(shoulder, elbow, wrist)
                    bacak_angle = calculate_angle(hip, knee, ankle)

                    cv2.putText(image, str(angle).split('.')[0], 
                                    tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 47, 47), 2, cv2.LINE_AA
                                        )
                    
                    cv2.putText(image, str(bacak_angle).split('.')[0], 
                                    tuple(np.multiply(knee, [640, 480]).astype(int)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 47, 47), 2, cv2.LINE_AA
                                        )
                    
                    if bacak_angle > 80 and bacak_angle < 150:
                        if angle > 150:
                            stage = "up"
                        if angle < 100 and stage == "up":
                            counter +=1
                            if counter==1:
                                startTime = datetime.now()
                                print(f"{startTime}")
                            print(counter)
                            stage = "down"   
                    else:
                        cv2.putText(image, 'BACAK ACISINI DOGRU YAPINIZ.', (190,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            except:
                pass

            cv2.rectangle(image, (0,0), (180,80), (245,117,16), -1)
            
            cv2.putText(image, 'Bench Press', (10,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        
            cv2.putText(image, stage, 
                        (40,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            
            if int(randomGoal) <= int(counter):
                
                cv2.putText(image, "Goal Completed. You can continue if you want",
                        (185,42), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                
                
            
            if (int(randomGoal) == int(counter)) and finishState:
                finishState = False
                finishTime = datetime.now()
                exerciseTime = finishTime - startTime
                print(f"yapılan süre: {exerciseTime.total_seconds()}")

                percent = round(((exerciseTime.total_seconds())*100) / (defaultExerciseTime * randomGoal))
                
                print(f"percent: {percent}")
                if percent > 100:
                    print("100 den büyük. yavaş yapıldı")

                    percent = percent - 100
                    pointLowerCount = percent // 10
                    print(f"pointLowerCount: {pointLowerCount}")
                    if pointLowerCount > 9:
                        point = 1
                    else:
                        point = 10 - pointLowerCount

                    print(f"Point: {point}")

                else:
                    print("100 den küçük. hızlı yapıldı")

                    percent = 100 - percent
                    pointLowerCount = percent // 10
                    print(f"pointLowerCount: {pointLowerCount}")
                    if pointLowerCount > 9:
                        point = 1
                    else:
                        point = 10 - pointLowerCount

                    print(f"Point: {point}")
                    

                

                
            cv2.putText(image, str(datetime.now()),
                        (185,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

            cv2.putText(image, "Goal: "+str(randomGoal), 
                        (10,52), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            
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

def chest_fly_machine():
    print("chest_fly_machine fonksiyonu çağrıldı")

def deadlift():
    print("deadlift fonksiyonu çağrıldı")

    randomGoal = generateRandomExerciseCount()
    print(f"random goal: {randomGoal}")

    defaultExerciseTime = 2.6

    startTime = datetime.now()
    finishTime = datetime.now()
    goalTime = float(randomGoal) * defaultExerciseTime
    finishState = True

    print(f"random goal: {randomGoal}")
    print(f"goal time: {goalTime}")


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
                    hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

                    bhip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    bknee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    bankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    
                    angle = calculate_angle(shoulder, hip, knee)

                    bacak_angle = calculate_angle(bhip, bknee, bankle)

                    cv2.putText(image, str(angle).split('.')[0], 
                                    tuple(np.multiply(hip, [640, 480]).astype(int)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )
                    
                    
                    if angle > 160:
                        stage = "up"
                    if angle < 110 and stage == "up":
                        counter +=1
                        if counter==1:
                            startTime = datetime.now()
                            print(f"{startTime}")
                        print(counter)
                        stage = "down"
                    

            except:
                pass

            cv2.rectangle(image, (0,0), (180,80), (245,117,16), -1)
            
            cv2.putText(image, 'Dead Lift', (10,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        
            cv2.putText(image, stage, 
                        (40,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            
            if int(randomGoal) <= int(counter):
                
                cv2.putText(image, "Goal Completed. You can continue if you want",
                        (185,42), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                
                
            
            if (int(randomGoal) == int(counter)) and finishState:
                finishState = False
                finishTime = datetime.now()
                exerciseTime = finishTime - startTime
                print(f"yapılan süre: {exerciseTime.total_seconds()}")

                percent = round(((exerciseTime.total_seconds())*100) / (defaultExerciseTime * randomGoal))
                
                print(f"percent: {percent}")
                if percent > 100:
                    print("100 den büyük. yavaş yapıldı")

                    percent = percent - 100
                    pointLowerCount = percent // 10
                    print(f"pointLowerCount: {pointLowerCount}")
                    if pointLowerCount > 9:
                        point = 1
                    else:
                        point = 10 - pointLowerCount

                    print(f"Point: {point}")

                else:
                    print("100 den küçük. hızlı yapıldı")

                    percent = 100 - percent
                    pointLowerCount = percent // 10
                    print(f"pointLowerCount: {pointLowerCount}")
                    if pointLowerCount > 9:
                        point = 1
                    else:
                        point = 10 - pointLowerCount

                    print(f"Point: {point}")
                    

                

                
            cv2.putText(image, str(datetime.now()),
                        (185,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

            cv2.putText(image, "Goal: "+str(randomGoal), 
                        (10,52), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            
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

def decline_bench_press():
    #yapılacak
    print("decline_bench_press fonksiyonu çağrıldı")

    randomGoal = generateRandomExerciseCount()
    print(f"random goal: {randomGoal}")

    defaultExerciseTime = 2.2

    startTime = datetime.now()
    finishTime = datetime.now()
    goalTime = float(randomGoal) * defaultExerciseTime
    finishState = True

    print(f"random goal: {randomGoal}")
    print(f"goal time: {goalTime}")

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
                    
                if (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility < 0.5) and (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility < 0.5):
                    cv2.putText(image, 'OMUZLAR ALGILANMADI', (190,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                    state = False

                if state:
                    if landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility > landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility:
                        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                        elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                        wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                        #xValue = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x
                        #yValue = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
                        #print("right daha net")
                    else:
                        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                        #xValue = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x
                        #yValue = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
                        #print("left daha net")

                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    
                    angle = calculate_angle(shoulder, elbow, wrist)
                    #bacak_angle = calculate_angle(hip, knee, ankle)

                    cv2.putText(image, str(angle).split('.')[0], 
                                    tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 47, 47), 2, cv2.LINE_AA
                                        )
                    
                    #cv2.putText(image, str(bacak_angle).split('.')[0], 
                    #                tuple(np.multiply(knee, [640, 480]).astype(int)), 
                    #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 47, 47), 2, cv2.LINE_AA
                    #                    )
                    
                    if angle > 150:
                        stage = "up"
                    if angle < 100 and stage == "up":
                        counter +=1
                        if counter==1:
                            startTime = datetime.now()
                            print(f"{startTime}")
                        print(counter)
                        stage = "down"   
                    
            except:
                pass

            cv2.rectangle(image, (0,0), (180,80), (245,117,16), -1)
            
            cv2.putText(image, 'Decline Bench Press', (10,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        
            cv2.putText(image, stage, 
                        (40,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            
            if int(randomGoal) <= int(counter):
                
                cv2.putText(image, "Goal Completed. You can continue if you want",
                        (185,42), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                
                
            
            if (int(randomGoal) == int(counter)) and finishState:
                finishState = False
                finishTime = datetime.now()
                exerciseTime = finishTime - startTime
                print(f"yapılan süre: {exerciseTime.total_seconds()}")

                percent = round(((exerciseTime.total_seconds())*100) / (defaultExerciseTime * randomGoal))
                
                print(f"percent: {percent}")
                if percent > 100:
                    print("100 den büyük. yavaş yapıldı")

                    percent = percent - 100
                    pointLowerCount = percent // 10
                    print(f"pointLowerCount: {pointLowerCount}")
                    if pointLowerCount > 9:
                        point = 1
                    else:
                        point = 10 - pointLowerCount

                    print(f"Point: {point}")

                else:
                    print("100 den küçük. hızlı yapıldı")

                    percent = 100 - percent
                    pointLowerCount = percent // 10
                    print(f"pointLowerCount: {pointLowerCount}")
                    if pointLowerCount > 9:
                        point = 1
                    else:
                        point = 10 - pointLowerCount

                    print(f"Point: {point}")
                    

                

                
            cv2.putText(image, str(datetime.now()),
                        (185,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

            cv2.putText(image, "Goal: "+str(randomGoal), 
                        (10,52), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            
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

def hammer_curl():
    print("hammer_curl fonksiyonu çağrıldı")

    randomGoal = generateRandomExerciseCount()
    print(f"random goal: {randomGoal}")

    defaultExerciseTime = 3.0

    startTime = datetime.now()
    finishTime = datetime.now()
    goalTime = float(randomGoal) * defaultExerciseTime
    finishState = True

    print(f"random goal: {randomGoal}")
    print(f"goal time: {goalTime}")


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

                if (landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].visibility < 0.5) and (landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility < 0.5):
                    cv2.putText(image, 'DIRSEKLER ALGILANMADI', (190,42), 
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
                    
                    
                    if angle > 100:
                        stage = "down"
                    if angle < 100 and stage == "down":
                        counter +=1
                        if counter==1:
                            startTime = datetime.now()
                            print(f"{startTime}")
                        print(counter)
                        stage = "up"   
                    

            except:
                pass

            cv2.rectangle(image, (0,0), (180,80), (245,117,16), -1)
            
            cv2.putText(image, 'Hammer Curl', (10,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        
            cv2.putText(image, stage, 
                        (40,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            
            if int(randomGoal) <= int(counter):
                
                cv2.putText(image, "Goal Completed. You can continue if you want",
                        (185,42), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                
                
            
            if (int(randomGoal) == int(counter)) and finishState:
                finishState = False
                finishTime = datetime.now()
                exerciseTime = finishTime - startTime
                print(f"yapılan süre: {exerciseTime.total_seconds()}")

                percent = round(((exerciseTime.total_seconds())*100) / (defaultExerciseTime * randomGoal))
                
                print(f"percent: {percent}")
                if percent > 100:
                    print("100 den büyük. yavaş yapıldı")

                    percent = percent - 100
                    pointLowerCount = percent // 10
                    print(f"pointLowerCount: {pointLowerCount}")
                    if pointLowerCount > 9:
                        point = 1
                    else:
                        point = 10 - pointLowerCount

                    print(f"Point: {point}")

                else:
                    print("100 den küçük. hızlı yapıldı")

                    percent = 100 - percent
                    pointLowerCount = percent // 10
                    print(f"pointLowerCount: {pointLowerCount}")
                    if pointLowerCount > 9:
                        point = 1
                    else:
                        point = 10 - pointLowerCount

                    print(f"Point: {point}")
                    

                

                
            cv2.putText(image, str(datetime.now()),
                        (185,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

            cv2.putText(image, "Goal: "+str(randomGoal), 
                        (10,52), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            
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

def hip_thrust():
    print("hip_thrust fonksiyonu çağrıldı")

    randomGoal = generateRandomExerciseCount()
    print(f"random goal: {randomGoal}")

    defaultExerciseTime = 2.8

    startTime = datetime.now()
    finishTime = datetime.now()
    goalTime = float(randomGoal) * defaultExerciseTime
    finishState = True

    print(f"random goal: {randomGoal}")
    print(f"goal time: {goalTime}")


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


                if (landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility < 0.5) and (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility < 0.5):
                    cv2.putText(image, 'DIZLER ALGILANMADI', (190,27), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                    state = False


                if state:
                    
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    
                    #angle = calculate_angle(shoulder, elbow, wrist)
                    bacak_angle = calculate_angle(shoulder, hip, knee)

                    cv2.putText(image, str(bacak_angle).split('.')[0], 
                                    tuple(np.multiply(hip, [640, 480]).astype(int)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 47, 47), 2, cv2.LINE_AA
                                        )
                    
                    #cv2.putText(image, str(bacak_angle).split('.')[0], 
                    #                tuple(np.multiply(knee, [640, 480]).astype(int)), 
                    #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 47, 47), 2, cv2.LINE_AA
                    #                    )
                    
                    if bacak_angle > 150:
                        stage = "up"
                    if bacak_angle < 100 and stage == "up":
                        counter +=1
                        if counter==1:
                            startTime = datetime.now()
                            print(f"{startTime}")
                        print(counter)
                        stage = "down"   
                    
            except:
                pass

            cv2.rectangle(image, (0,0), (180,80), (245,117,16), -1)
            
            cv2.putText(image, 'Hip Thrust', (10,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        
            cv2.putText(image, stage, 
                        (40,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            
            if int(randomGoal) <= int(counter):
                
                cv2.putText(image, "Goal Completed. You can continue if you want",
                        (185,42), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                
                
            
            if (int(randomGoal) == int(counter)) and finishState:
                finishState = False
                finishTime = datetime.now()
                exerciseTime = finishTime - startTime
                print(f"yapılan süre: {exerciseTime.total_seconds()}")

                percent = round(((exerciseTime.total_seconds())*100) / (defaultExerciseTime * randomGoal))
                
                print(f"percent: {percent}")
                if percent > 100:
                    print("100 den büyük. yavaş yapıldı")

                    percent = percent - 100
                    pointLowerCount = percent // 10
                    print(f"pointLowerCount: {pointLowerCount}")
                    if pointLowerCount > 9:
                        point = 1
                    else:
                        point = 10 - pointLowerCount

                    print(f"Point: {point}")

                else:
                    print("100 den küçük. hızlı yapıldı")

                    percent = 100 - percent
                    pointLowerCount = percent // 10
                    print(f"pointLowerCount: {pointLowerCount}")
                    if pointLowerCount > 9:
                        point = 1
                    else:
                        point = 10 - pointLowerCount

                    print(f"Point: {point}")
                    

                

                
            cv2.putText(image, str(datetime.now()),
                        (185,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

            cv2.putText(image, "Goal: "+str(randomGoal), 
                        (10,52), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            
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

def incline_bench_press():
    print("incline_bench_press fonksiyonu çağrıldı")

    randomGoal = generateRandomExerciseCount()
    print(f"random goal: {randomGoal}")

    defaultExerciseTime = 2.2

    startTime = datetime.now()
    finishTime = datetime.now()
    goalTime = float(randomGoal) * defaultExerciseTime
    finishState = True

    print(f"random goal: {randomGoal}")
    print(f"goal time: {goalTime}")


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

                if (landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].visibility < 0.5) and (landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility < 0.5):
                    cv2.putText(image, 'DIRSEKLER ALGILANMADI', (190,42), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                    state = False
                    
                if (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility < 0.5) and (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility < 0.5):
                    cv2.putText(image, 'OMUZLAR ALGILANMADI', (190,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                    state = False

                if state:
                    if landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility > landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility:
                        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                        elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                        wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                        #xValue = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x
                        #yValue = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
                        #print("right daha net")
                    else:
                        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                        #xValue = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x
                        #yValue = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
                        #print("left daha net")

                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    
                    angle = calculate_angle(shoulder, elbow, wrist)
                    #bacak_angle = calculate_angle(hip, knee, ankle)

                    cv2.putText(image, str(angle).split('.')[0], 
                                    tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 47, 47), 2, cv2.LINE_AA
                                        )
                    
                    #cv2.putText(image, str(bacak_angle).split('.')[0], 
                    #                tuple(np.multiply(knee, [640, 480]).astype(int)), 
                    #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 47, 47), 2, cv2.LINE_AA
                    #                    )
                    
                    if angle > 150:
                        stage = "up"
                    if angle < 100 and stage == "up":
                        counter +=1
                        if counter==1:
                            startTime = datetime.now()
                            print(f"{startTime}")
                        print(counter)
                        stage = "down"   
                    
            except:
                pass

            cv2.rectangle(image, (0,0), (180,80), (245,117,16), -1)
            
            cv2.putText(image, 'Incline Bench Press', (10,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        
            cv2.putText(image, stage, 
                        (40,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            
            if int(randomGoal) <= int(counter):
                
                cv2.putText(image, "Goal Completed. You can continue if you want",
                        (185,42), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                
                
            
            if (int(randomGoal) == int(counter)) and finishState:
                finishState = False
                finishTime = datetime.now()
                exerciseTime = finishTime - startTime
                print(f"yapılan süre: {exerciseTime.total_seconds()}")

                percent = round(((exerciseTime.total_seconds())*100) / (defaultExerciseTime * randomGoal))
                
                print(f"percent: {percent}")
                if percent > 100:
                    print("100 den büyük. yavaş yapıldı")

                    percent = percent - 100
                    pointLowerCount = percent // 10
                    print(f"pointLowerCount: {pointLowerCount}")
                    if pointLowerCount > 9:
                        point = 1
                    else:
                        point = 10 - pointLowerCount

                    print(f"Point: {point}")

                else:
                    print("100 den küçük. hızlı yapıldı")

                    percent = 100 - percent
                    pointLowerCount = percent // 10
                    print(f"pointLowerCount: {pointLowerCount}")
                    if pointLowerCount > 9:
                        point = 1
                    else:
                        point = 10 - pointLowerCount

                    print(f"Point: {point}")
                    

                

                
            cv2.putText(image, str(datetime.now()),
                        (185,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

            cv2.putText(image, "Goal: "+str(randomGoal), 
                        (10,52), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            
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

def lat_pulldown():
    print("lat_pulldown fonksiyonu çağrıldı")

    randomGoal = generateRandomExerciseCount()
    print(f"random goal: {randomGoal}")

    defaultExerciseTime = 5.0

    startTime = datetime.now()
    finishTime = datetime.now()
    goalTime = float(randomGoal) * defaultExerciseTime
    finishState = True

    print(f"random goal: {randomGoal}")
    print(f"goal time: {goalTime}")


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
                    
                if (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility < 0.5) and (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility < 0.5):
                    cv2.putText(image, 'OMUZLAR ALGILANMADI', (190,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                    state = False

                if state:
                    if landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility > landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility:
                        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                        elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                        wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                        #xValue = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x
                        #yValue = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
                        #print("right daha net")
                    else:
                        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                        #xValue = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x
                        #yValue = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
                        #print("left daha net")



                    if landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility > landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility:
                        hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                        knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                        ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                    else:
                        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                    angle = calculate_angle(shoulder, elbow, wrist)
                    bacak_angle = calculate_angle(hip, knee, ankle)

                    cv2.putText(image, str(angle).split('.')[0], 
                                    tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 47, 47), 2, cv2.LINE_AA
                                        )
                    
                    cv2.putText(image, str(bacak_angle).split('.')[0], 
                                    tuple(np.multiply(knee, [640, 480]).astype(int)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 47, 47), 2, cv2.LINE_AA
                                        )
                    
                    if bacak_angle < 100:
                        if angle > 150:
                            stage = "up"
                        if angle < 100 and stage == "up":
                            counter +=1
                            if counter==1:
                                startTime = datetime.now()
                                print(f"{startTime}")
                            print(counter)
                            stage = "down"   
                    else:
                        cv2.putText(image, 'BACAK ACISINI DOGRU YAPINIZ.', (190,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            except:
                pass

            cv2.rectangle(image, (0,0), (180,80), (245,117,16), -1)
            
            cv2.putText(image, 'Lat Pulldown', (10,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        
            cv2.putText(image, stage, 
                        (40,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            
            if int(randomGoal) <= int(counter):
                
                cv2.putText(image, "Goal Completed. You can continue if you want",
                        (185,42), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                
                
            
            if (int(randomGoal) == int(counter)) and finishState:
                finishState = False
                finishTime = datetime.now()
                exerciseTime = finishTime - startTime
                print(f"yapılan süre: {exerciseTime.total_seconds()}")

                percent = round(((exerciseTime.total_seconds())*100) / (defaultExerciseTime * randomGoal))
                
                print(f"percent: {percent}")
                if percent > 100:
                    print("100 den büyük. yavaş yapıldı")

                    percent = percent - 100
                    pointLowerCount = percent // 10
                    print(f"pointLowerCount: {pointLowerCount}")
                    if pointLowerCount > 9:
                        point = 1
                    else:
                        point = 10 - pointLowerCount

                    print(f"Point: {point}")

                else:
                    print("100 den küçük. hızlı yapıldı")

                    percent = 100 - percent
                    pointLowerCount = percent // 10
                    print(f"pointLowerCount: {pointLowerCount}")
                    if pointLowerCount > 9:
                        point = 1
                    else:
                        point = 10 - pointLowerCount

                    print(f"Point: {point}")
                    

                

                
            cv2.putText(image, str(datetime.now()),
                        (185,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

            cv2.putText(image, "Goal: "+str(randomGoal), 
                        (10,52), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            
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

def lateral_raises():
    print("lateral_raises fonksiyonu çağrıldı")

    randomGoal = generateRandomExerciseCount()
    print(f"random goal: {randomGoal}")

    defaultExerciseTime = 3.7

    startTime = datetime.now()
    finishTime = datetime.now()
    goalTime = float(randomGoal) * defaultExerciseTime
    finishState = True

    print(f"random goal: {randomGoal}")
    print(f"goal time: {goalTime}")

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


                if (landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility < 0.5) and (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility < 0.5):
                    cv2.putText(image, 'DIZLER ALGILANMADI', (190,27), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                    state = False


                if state:
                    
                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    
                    angle = calculate_angle(elbow, shoulder, hip)
                    #bacak_angle = calculate_angle(shoulder, hip, knee)

                    cv2.putText(image, str(angle).split('.')[0], 
                                    tuple(np.multiply(shoulder, [640, 480]).astype(int)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 47, 47), 2, cv2.LINE_AA
                                        )
                    
                    if angle > 60:
                        stage = "up"
                    if angle < 50 and stage == "up":
                        counter +=1
                        if counter==1:
                            startTime = datetime.now()
                            print(f"{startTime}")
                        print(counter)
                        stage = "down"   
                    
            except:
                pass

            cv2.rectangle(image, (0,0), (180,80), (245,117,16), -1)
            
            cv2.putText(image, 'Lateral Raises', (10,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        
            cv2.putText(image, stage, 
                        (40,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            
            if int(randomGoal) <= int(counter):
                
                cv2.putText(image, "Goal Completed. You can continue if you want",
                        (185,42), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                
                
            
            if (int(randomGoal) == int(counter)) and finishState:
                finishState = False
                finishTime = datetime.now()
                exerciseTime = finishTime - startTime
                print(f"yapılan süre: {exerciseTime.total_seconds()}")

                percent = round(((exerciseTime.total_seconds())*100) / (defaultExerciseTime * randomGoal))
                
                print(f"percent: {percent}")
                if percent > 100:
                    print("100 den büyük. yavaş yapıldı")

                    percent = percent - 100
                    pointLowerCount = percent // 10
                    print(f"pointLowerCount: {pointLowerCount}")
                    if pointLowerCount > 9:
                        point = 1
                    else:
                        point = 10 - pointLowerCount

                    print(f"Point: {point}")

                else:
                    print("100 den küçük. hızlı yapıldı")

                    percent = 100 - percent
                    pointLowerCount = percent // 10
                    print(f"pointLowerCount: {pointLowerCount}")
                    if pointLowerCount > 9:
                        point = 1
                    else:
                        point = 10 - pointLowerCount

                    print(f"Point: {point}")
                    

                

                
            cv2.putText(image, str(datetime.now()),
                        (185,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

            cv2.putText(image, "Goal: "+str(randomGoal), 
                        (10,52), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            
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

def leg_extension():
    print("leg_extension fonksiyonu çağrıldı")

    randomGoal = generateRandomExerciseCount()
    print(f"random goal: {randomGoal}")

    defaultExerciseTime = 3.0

    startTime = datetime.now()
    finishTime = datetime.now()
    goalTime = float(randomGoal) * defaultExerciseTime
    finishState = True

    print(f"random goal: {randomGoal}")
    print(f"goal time: {goalTime}")

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


                if (landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility < 0.5) and (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility < 0.5):
                    cv2.putText(image, 'DIZLER ALGILANMADI', (190,27), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                    state = False


                if state:
                    
                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    

                    angle = calculate_angle(shoulder, hip, knee)

                    bacak_angle = calculate_angle(hip, knee, ankle)

                    cv2.putText(image, str(angle).split('.')[0], 
                                    tuple(np.multiply(hip, [640, 480]).astype(int)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 47, 47), 2, cv2.LINE_AA
                                        )
                    
                    cv2.putText(image, str(bacak_angle).split('.')[0], 
                                    tuple(np.multiply(knee, [640, 480]).astype(int)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 47, 47), 2, cv2.LINE_AA
                                        )
                    

                    if angle > 80 and angle < 110:
                        if bacak_angle > 150:
                            stage = "up"
                        if bacak_angle < 100 and stage == "up":
                            counter +=1
                            if counter==1:
                                startTime = datetime.now()
                                print(f"{startTime}")
                            print(counter)
                            stage = "down"   
                    else:
                        cv2.putText(image, 'LUTFEN DIK DURUNUZ', (190,27), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                    
            except:
                pass

            cv2.rectangle(image, (0,0), (180,80), (245,117,16), -1)
            
            cv2.putText(image, 'Leg Extensions', (10,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        
            cv2.putText(image, stage, 
                        (40,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            
            if int(randomGoal) <= int(counter):
                
                cv2.putText(image, "Goal Completed. You can continue if you want",
                        (185,42), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                
                
            
            if (int(randomGoal) == int(counter)) and finishState:
                finishState = False
                finishTime = datetime.now()
                exerciseTime = finishTime - startTime
                print(f"yapılan süre: {exerciseTime.total_seconds()}")

                percent = round(((exerciseTime.total_seconds())*100) / (defaultExerciseTime * randomGoal))
                
                print(f"percent: {percent}")
                if percent > 100:
                    print("100 den büyük. yavaş yapıldı")

                    percent = percent - 100
                    pointLowerCount = percent // 10
                    print(f"pointLowerCount: {pointLowerCount}")
                    if pointLowerCount > 9:
                        point = 1
                    else:
                        point = 10 - pointLowerCount

                    print(f"Point: {point}")

                else:
                    print("100 den küçük. hızlı yapıldı")

                    percent = 100 - percent
                    pointLowerCount = percent // 10
                    print(f"pointLowerCount: {pointLowerCount}")
                    if pointLowerCount > 9:
                        point = 1
                    else:
                        point = 10 - pointLowerCount

                    print(f"Point: {point}")
                    

                

                
            cv2.putText(image, str(datetime.now()),
                        (185,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

            cv2.putText(image, "Goal: "+str(randomGoal), 
                        (10,52), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            
            
            
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

def leg_raises():
    print("leg_raises fonksiyonu çağrıldı")

    randomGoal = generateRandomExerciseCount()
    print(f"random goal: {randomGoal}")

    defaultExerciseTime = 2.0

    startTime = datetime.now()
    finishTime = datetime.now()
    goalTime = float(randomGoal) * defaultExerciseTime
    finishState = True

    print(f"random goal: {randomGoal}")
    print(f"goal time: {goalTime}")


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


                if (landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility < 0.5) and (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility < 0.5):
                    cv2.putText(image, 'DIZLER ALGILANMADI', (190,27), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                    state = False


                if state:
                    
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    

                    angle = calculate_angle(shoulder, hip, knee)

                    cv2.putText(image, str(angle).split('.')[0], 
                                    tuple(np.multiply(hip, [640, 480]).astype(int)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 47, 47), 2, cv2.LINE_AA
                                        )

                    

                    if angle < 120:
                        stage = "up"
                    if angle > 140 and stage == "up":
                        counter +=1
                        if counter==1:
                            startTime = datetime.now()
                            print(f"{startTime}")
                        print(counter)
                        stage = "down"  

                    
            except:
                pass

            cv2.rectangle(image, (0,0), (180,80), (245,117,16), -1)
            
            cv2.putText(image, 'Leg Raises', (10,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        
            cv2.putText(image, stage, 
                        (40,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            
            if int(randomGoal) <= int(counter):
                
                cv2.putText(image, "Goal Completed. You can continue if you want",
                        (185,42), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                
                
            
            if (int(randomGoal) == int(counter)) and finishState:
                finishState = False
                finishTime = datetime.now()
                exerciseTime = finishTime - startTime
                print(f"yapılan süre: {exerciseTime.total_seconds()}")

                percent = round(((exerciseTime.total_seconds())*100) / (defaultExerciseTime * randomGoal))
                
                print(f"percent: {percent}")
                if percent > 100:
                    print("100 den büyük. yavaş yapıldı")

                    percent = percent - 100
                    pointLowerCount = percent // 10
                    print(f"pointLowerCount: {pointLowerCount}")
                    if pointLowerCount > 9:
                        point = 1
                    else:
                        point = 10 - pointLowerCount

                    print(f"Point: {point}")

                else:
                    print("100 den küçük. hızlı yapıldı")

                    percent = 100 - percent
                    pointLowerCount = percent // 10
                    print(f"pointLowerCount: {pointLowerCount}")
                    if pointLowerCount > 9:
                        point = 1
                    else:
                        point = 10 - pointLowerCount

                    print(f"Point: {point}")
                    

                

                
            cv2.putText(image, str(datetime.now()),
                        (185,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

            cv2.putText(image, "Goal: "+str(randomGoal), 
                        (10,52), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            
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

    randomGoal = generateRandomExerciseCount()
    print(f"random goal: {randomGoal}")

    defaultExerciseTime = 2.0

    startTime = datetime.now()
    finishTime = datetime.now()
    goalTime = float(randomGoal) * defaultExerciseTime
    finishState = True

    print(f"random goal: {randomGoal}")
    print(f"goal time: {goalTime}")


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


                if (landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility < 0.5) and (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility < 0.5):
                    cv2.putText(image, 'DIZLER ALGILANMADI', (190,27), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                    state = False


                if state:
                    
                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    
                    

                    angle = calculate_angle(hip, knee, ankle)

                    cv2.putText(image, str(angle).split('.')[0], 
                                    tuple(np.multiply(knee, [640, 480]).astype(int)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 47, 47), 2, cv2.LINE_AA
                                        )

                    

                    if angle < 160:
                        stage = "down"
                    if angle > 165 and stage == "down":
                        counter +=1
                        if counter==1:
                            startTime = datetime.now()
                            print(f"{startTime}")
                        print(counter)
                        stage = "up"  

                    
            except:
                pass

            cv2.rectangle(image, (0,0), (180,80), (245,117,16), -1)
            
            cv2.putText(image, 'Squat', (10,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        
            cv2.putText(image, stage, 
                        (40,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            
            if int(randomGoal) <= int(counter):
                
                cv2.putText(image, "Goal Completed. You can continue if you want",
                        (185,42), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                
                
            
            if (int(randomGoal) == int(counter)) and finishState:
                finishState = False
                finishTime = datetime.now()
                exerciseTime = finishTime - startTime
                print(f"yapılan süre: {exerciseTime.total_seconds()}")

                percent = round(((exerciseTime.total_seconds())*100) / (defaultExerciseTime * randomGoal))
                
                print(f"percent: {percent}")
                if percent > 100:
                    print("100 den büyük. yavaş yapıldı")

                    percent = percent - 100
                    pointLowerCount = percent // 10
                    print(f"pointLowerCount: {pointLowerCount}")
                    if pointLowerCount > 9:
                        point = 1
                    else:
                        point = 10 - pointLowerCount

                    print(f"Point: {point}")

                else:
                    print("100 den küçük. hızlı yapıldı")

                    percent = 100 - percent
                    pointLowerCount = percent // 10
                    print(f"pointLowerCount: {pointLowerCount}")
                    if pointLowerCount > 9:
                        point = 1
                    else:
                        point = 10 - pointLowerCount

                    print(f"Point: {point}")
                    

                

                
            cv2.putText(image, str(datetime.now()),
                        (185,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

            cv2.putText(image, "Goal: "+str(randomGoal), 
                        (10,52), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            
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


def t_bar_row():
    print("t_bar_row fonksiyonu çağrıldı")

def tricep_dips():
    print("tricep_dips fonksiyonu çağrıldı")

def tricep_pushdown():
    print("tricep_pushdown fonksiyonu çağrıldı")
