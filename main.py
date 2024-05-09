import tkinter as tk
from tkinter import ttk
import exercises_function
from anglecalculator import calculate_angle
import cv2
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from keras.optimizers import SGD, Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils import to_categorical
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
from sklearn.utils import shuffle
from keras import utils
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
import keras
from keras.layers import Dense, Activation, Dropout, Reshape, Permute, Conv2D
import math
import numpy as np
import mediapipe as mp
import asyncio


classes = {
     0:  "barbell_biceps_curl",
     1:  "bench_press",
     2:  "chest_fly_machine",
     3:  "deadlift",
     4:  "decline_bench_press",
     5:  "hammer_curl",
     6:  "hip_thrust",
     7:  "incline_bench_press",
     8:  "lat_pulldown",
     9:  "lateral_raises",
     10: "leg_extension",
     11: "leg_raises",
     12: "plank",
     13: "pull_up",
     14: "push_up",
     15: "romanian_deadlift",
     16: "russian_twist",
     17: "shoulder_press",
     18: "squat",
     19: "t_bar_row",
     20: "tricep_dips",
     21: "tricep_pushdown",
}


model = load_model('workout_model_.h5')
framess = []
def predict():
    selected_index = combo_box.current()
    fonksiyon = classes[selected_index]
    #exercises_function.push_up()
    fonksiyon = getattr(exercises_function,fonksiyon)
    fonksiyon()

def choose_exercises_with_camera():
    global counter 
    global framess
    stage = None

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(0)

    liste = []

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
            
                results = pose.process(image)
            
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if framess is not None:
                    if(len(framess) < 111):
                        framess.append(frame)
                    else:
                        #print(len(framess[:,:,10]))
                        for i in range(0, len(framess), 10):
                            frame = framess[i]

                            sample_image = cv2.resize(frame, (256, 256))
                            sample_image = np.expand_dims(sample_image, axis=0) 

                            predictions = model.predict(sample_image)
                            print(str(i)+" predictions:")
                            print(predictions)
                            print(type(predictions))
                            predicted_class = np.argmax(predictions)
                            print(str(i)+" predict:")
                            print(predicted_class)

                            liste.append(predicted_class)

                        for eleman in set(liste):
                            print(f"{eleman}: {liste.count(eleman)}")

                        en_fazla_eleman = max(set(liste), key=liste.count)
                        en_fazla_sayi = liste.count(en_fazla_eleman)
                        print(classes[en_fazla_eleman])

                        fonksiyon = classes[en_fazla_eleman]
                        fonksiyon = getattr(exercises_function,fonksiyon)
                        cap.release()
                        cv2.destroyAllWindows()
                        #exercises_function.push_up()
                        fonksiyon()

                        framess = None

                if (framess is None) or (len(framess) == 111):
                    cv2.putText(image, 'HAREKETLER ANALIZ EDILIYOR. BEKLEYINIZ', (175,20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                else:
                    cv2.putText(image, '5 SANIYE BOYUNCA AYNI HAREKETI YAPINIZ', (175,20), 
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
    framess = []

root = tk.Tk()
root.title("Egzersiz Tahmini")
root.geometry("300x150")
root.lift()


camera_button = tk.Button(root, text="Choose Exercises with Camera", command=choose_exercises_with_camera)
camera_button.pack(pady=10)

combo_box = ttk.Combobox(root, state = "readonly", values=[i for i in classes.values()])
combo_box.pack(pady=10)


for a in classes.values():
    print(str(a))

predict_button = tk.Button(root, text="Predict", command=predict)
predict_button.pack(pady=10)

root.mainloop()
