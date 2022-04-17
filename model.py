import numpy as np 
import pandas as pd
import cv2
import random
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten,Dense
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
import tensorflow as tf
import os

dataset=[]
model=None
image_size=224
labels = []
dataset = []
x_train=[]
x_test=[]
y_train=[]
y_test=[]
y_pred=[]
dataset_dir = "./dataset/preprocessed_images"
input_dir="./static/ImageUploads"

def data_preprocessing():
    print("*************** Data Preprocessing Started ***************")
    global x_train 
    global x_test 
    global y_train 
    global y_test
    global image_size
    global cataract
    global normal
    global dataset
    global dataset_dir

    df = pd.read_csv("./dataset/full_df.csv")
    # df.head()
    def has_cataract(text):
        if "cataract" in text:
            return 1
        else:
            return 0
    df["left_cataract"] = df["Left-Diagnostic Keywords"].apply(lambda x: has_cataract(x))
    df["right_cataract"] = df["Right-Diagnostic Keywords"].apply(lambda x: has_cataract(x))

    left_cataract = df.loc[(df.C ==1) & (df.left_cataract == 1)]["Left-Fundus"].values
    right_cataract = df.loc[(df.C ==1) & (df.right_cataract == 1)]["Right-Fundus"].values

    print("Number of images in left cataract: {}".format(len(left_cataract)))
    print("Number of images in right cataract: {}".format(len(right_cataract)))

    left_normal = df.loc[(df.C ==0) & (df["Left-Diagnostic Keywords"] == "normal fundus")]["Left-Fundus"].sample(250,random_state=42).values
    right_normal = df.loc[(df.C ==0) & (df["Right-Diagnostic Keywords"] == "normal fundus")]["Right-Fundus"].sample(250,random_state=42).values

    cataract = np.concatenate((left_cataract,right_cataract),axis=0)
    normal = np.concatenate((left_normal,right_normal),axis=0)

    print(len(cataract),len(normal))

    def create_dataset(image_category,label):
        global image_size
        for img in tqdm(image_category):
            image_path = os.path.join(dataset_dir,img)
            try:
                image = cv2.imread(image_path,cv2.IMREAD_COLOR)
                image = cv2.resize(image,(image_size,image_size))

            except:
                continue
            
            dataset.append([np.array(image),np.array(label)])
        random.shuffle(dataset)
        return dataset
    dataset = create_dataset(cataract,1)
    dataset = create_dataset(normal,0)
    x = np.array([i[0] for i in dataset]).reshape(-1,image_size,image_size,3)
    y = np.array([i[1] for i in dataset])

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
    print("*************** Data Preprocessing Ended ***************")


def save_model():
    print("*************** Saving Model ***************")

    global model
    model.save('saved_model/cataract_predict')
    print("*************** Model Saved Succesfully ***************")


def load_model():
    print("*************** Loading Existing Model Started ***************")

    if not os.path.exists('saved_model/cataract_predict') or len(os.listdir('saved_model/cataract_predict') ) == 0:
        print("Training Model")
        data_preprocessing()
        train_model()
    else:
        print("Loading Existing Model")
        global model
        model = tf.keras.models.load_model('saved_model/cataract_predict')
        model.summary()
    print("*************** Loading Existing Model Ended ***************")
    


def train_model():
    print("*************** Training Model Started ***************")

    global model
    global x_train 
    global x_test 
    global y_train 
    global y_test
    global image_size

    vgg = VGG19(weights="imagenet",include_top = False,input_shape=(image_size,image_size,3))

    for layer in vgg.layers:
        layer.trainable = False
    model = Sequential()
    model.add(vgg)
    model.add(Flatten())
    model.add(Dense(1,activation="sigmoid"))

    model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
    checkpoint = ModelCheckpoint("vgg19.h5",monitor="val_acc",verbose=1,save_best_only=True,
                             save_weights_only=False,period=1)
    earlystop = EarlyStopping(monitor="val_acc",patience=5,verbose=1)

    history = model.fit(x_train,y_train,batch_size=32,epochs=1,validation_data=(x_test,y_test),
                    verbose=1,callbacks=[checkpoint,earlystop])
    save_model()
    print("*************** Training Model Ended ***************")



def data_evaluate():
    print("*************** Data Evaluation Started ***************")

    if model==None:
        load_model()
    loss,accuracy = model.evaluate(x_test,y_test)
    print("loss:",loss)
    print("Accuracy:",accuracy)
    # accuracy_score(y_test,y_pred)
    print("*************** Data Evaluation Ended ***************")

    

def predict(filename=[]):
    global input_dir
    global image_size
    global y_pred
    inputDataset=[]
    print("*************** Data Prediction Started ***************")

    if model==None:
        load_model()
    for img in tqdm(filename):
        image_path = os.path.join(input_dir,img)
        try:
            image = cv2.imread(image_path,cv2.IMREAD_COLOR)
            image = cv2.resize(image,(image_size,image_size))
        except:
            filename.pop(img)
            continue    
        
        inputDataset.append([np.array(image) ,np.array(1)])
    _x = np.array([i[0] for i in inputDataset]).reshape(-1,image_size,image_size,3)
    _y = np.array([i[1] for i in inputDataset])

    prediction = (model.predict(_x) > 0.5).astype("int32")

    for i in range(len(filename)):
        print("Result", filename[i], ": ", prediction[i])
    print("*************** Data Prediction Ended ***************")
    return (filename,prediction)

if __name__ == "__main__":
    predict(["0_left.jpg", "0_right.jpg", "1_left.jpg", "1_right.jpg", "2_left.jpg", "2_right.jpg", "24_left.jpg", "24_right.jpg"])

