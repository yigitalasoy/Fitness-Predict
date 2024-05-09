import exercises_function

import matplotlib.pyplot as plt
import cv2
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from keras.optimizers import SGD, Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils import to_categorical
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
from sklearn.utils import shuffle
from keras import utils
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
import keras
from keras.layers import Dense, Activation, Dropout, Reshape, Permute, Conv2D
from sklearn.utils import shuffle
import cv2
import math
import numpy as np


classes = {
     0:  exercises_function.barbell_biceps_curl,
     1:  exercises_function.bench_press,
     2:  exercises_function.chest_fly_machine,
     3:  exercises_function.deadlift,
     4:  exercises_function.decline_bench_press,
     5:  exercises_function.hammer_curl,
     6:  exercises_function.hip_thrust,
     7:  exercises_function.incline_bench_press,
     8:  exercises_function.lat_pulldown,
     9:  exercises_function.lateral_raises,
     10: exercises_function.leg_extension,
     11: exercises_function.leg_raises,
     12: exercises_function.plank,
     13: exercises_function.pull_up,
     14: exercises_function.push_up,
     15: exercises_function.romanian_deadlift,
     16: exercises_function.russian_twist,
     17: exercises_function.shoulder_press,
     18: exercises_function.squat,
     19: exercises_function.t_bar_row,
     20: exercises_function.tricep_dips,
     21: exercises_function.tricep_pushdown,
}




model = load_model('workout_model_.h5')

cap = cv2.VideoCapture(0)

if cap.isOpened():
    ret, frame = cap.read()

    if ret:
        #cv2.imwrite('ornek_goruntu.png', frame)

        img = cv2.imread('Screenshot_1.png')
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
        contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        x,y,w,h = cv2.boundingRect(cnt)
        crop = frame[y:y+h,x:x+w]
        
        hedef_genislik = 256
        hedef_yukseklik = 256

        sample_image = cv2.resize(crop, (hedef_genislik, hedef_yukseklik), interpolation=cv2.INTER_AREA)
        cv2.imwrite('test.png',sample_image)
        sample_image = np.expand_dims(sample_image, axis=0)
        cap.release()
        

    #if ret:
    #    cv2.imwrite('sofwinresframe.png',frame)
    #    sample_image = cv2.resize(frame, (256, 256))
    #    cv2.imwrite('sofwinressample.png',sample_image)
    #    cv2.imwrite('ornek_goruntu.png', sample_image)
    #    sample_image = np.expand_dims(sample_image, axis=0) 
    #    cap.release()
    else:
        print("Kare alınamadı.")
else:
    print("Kamera başlatılamadı.")


predictions = model.predict(sample_image)
print("predictions:")
print(predictions)
print(type(predictions))
predicted_class = np.argmax(predictions)
print("predict:")
print(predicted_class)



print("Tahmin edilen sınıf:", classes[predicted_class])

fonksiyon = classes[predicted_class]
#exercises_function.push_up()
fonksiyon()




'''
kamera açılcak,
kullanıcı hareketi yapıcak,
o hareketi model ile tanımlayacağız,
tanımlanılan modelin python fonksiyonu çağırılacak
rastgele bir sayı belirlenicek ve kullanıcı o sayıya kadar
hareketi yapacak.
sayaç ve yapılan hareket eşit olduğunda tekrar başa dönücek
'''