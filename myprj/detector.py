## Lab2

import numpy as np
import cv2
import matplotlib
import pickle
import h5py
import glob
import time
from random import shuffle
from collections import Counter
from sklearn.model_selection import train_test_split
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import keras
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,Dropout,Softmax
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D,LeakyReLU,ReLU, AveragePooling2D
from keras.optimizers import SGD, Adam,RMSprop,Adagrad,Adadelta
from tensorflow.keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

TF_ENABLE_DEPRECATION_WARINGS=1

test_path = "Lab1/Dataset/F/"

def get_list(test_path):
    all_name = os.listdir(test_path)
    return all_name
def prepare(filepath):
    IMG_SIZE=32
    img_array=cv2.imread(filepath)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    img_array = img_array/255.0
    new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
    return new_array.reshape(-1,IMG_SIZE,IMG_SIZE,3)


def main():

    cnt = 0
    cnt_R = 0
    cnt_L = 0
    cnt_F = 0
    cnt_B = 0
    total_time = 0
    for i in get_list(test_path):
        if ".jpg" in i:
            cnt = cnt + 1
            time_classifier_i = time.time()

            preds=model.predict(prepare(test_path+i))

            time_classifier_o = time.time() - time_classifier_i 
            
            if cnt != 1:
                total_time = total_time + time_classifier_o
            print("photo " + i + " spent: "+ str(time_classifier_o)+"\n")
           
            predict_max = np.argmax(preds, -1)

            predict_max_int = int(predict_max[0])

            if predict_max_int == 0:
                cnt_R = cnt_R + 1
                print("photo " + i + " = Face_R\n\n\n")
            elif predict_max_int == 1:
                cnt_L = cnt_L + 1
                print("photo " + i + " = Face_L\n\n\n")
            elif predict_max_int == 2:
                cnt_F = cnt_F + 1
                print("photo " + i + " = Face_F\n\n\n")
            else:
                cnt_B = cnt_B + 1
                print("photo " + i + " = Face_B\n\n\n")
                
    print("cnt = ",cnt)
    print("cnt_R = ",cnt_R)
    print("cnt_L = ",cnt_L)
    print("cnt_F = ",cnt_F)
    print("cnt_B = ",cnt_B) 
    print("total_time = ",total_time)
    print("average_time = ",total_time / int(cnt-1))


if __name__ == '__main__': 
    model= keras.models.load_model("Lab1/classifier_lab0811.h5")

    main()

