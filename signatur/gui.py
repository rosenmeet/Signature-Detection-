#................starting gui librarie application ..........................
from tkinter import *
import tkinter as tk
from tkinter import messagebox
import datetime as dt
import time
from time import strftime
import os
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk
# ..............End Gui libraries............
#...............start Backend Prediction.......
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import PIL
from tensorflow.keras.preprocessing import image_dataset_from_directory
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import optimizers, losses, callbacks
from keras.models import Sequential
from tensorflow import keras
import tensorflow as tf
#...............end libraries backen.........
class Signature_detection:
    def __init__(self,root):
        self.root=root
        self,root.configure(bg="red")
        self.root.title("Signature_detection.com")
        self.root.geometry("1100x680+200+30")
        self.root.maxsize(width=700,height=700)
        self.root.minsize(width=700,height=700)
        
        self.FilePath=StringVar()
        #.........................................Heading Frame......................................................................................
        T_frame=Frame(self.root,bd=10,relief=RIDGE,bg="white")
        T_frame.place(x=10,y=10,width=680,height=60)
        I_lb1=Label(T_frame,text="Machine Learning Signature Detectioon",font=("times new roman",15,"bold"),bg='white',fg='black',bd=0).place(x=120,y=7)      
#....................................middle frame..........................................................................................................
        M_frame=Frame(self.root,bd=10,relief=RIDGE,bg="white")
        M_frame.place(x=100,y=90,width=500,height=400)
        
        self.imglabel=Label(M_frame)
        self.imglabel.place(x=140,y=10,width=200,height=200)
                
        self.CN=Entry(M_frame,textvariable=self.FilePath,font=("times new roman",15),bg="#ECECEC",bd=5,relief=GROOVE)
        self.CN.place(x=50,y=220,width=380,height=50)
        
        self.btn_Clear=Button(M_frame,command=self.upload_img,text="Upload Image",font=("times new roman",17),bg="blue",activebackground='midnight blue',fg="white",activeforeground='white',cursor='hand2')
        self.btn_Clear.place(x=130,y=290,width=200,height=50)
#.....................button frame...........................................
        E_frame=Frame(self.root,bd=10,relief=RIDGE,bg="white")
        E_frame.place(x=40,y=500,width=600,height=100)
        
        self.btn_Pred=Button(E_frame,command=self.prediction,text="Predict",font=("times new roman",17),bg="blue",activebackground='midnight blue',fg="white",activeforeground='white',cursor='hand2')
        self.btn_Pred.place(x=60,y=15,width=230,height=50)
        
        self.btn_Clear=Button(E_frame,command=self.exit,text="Clear Now",font=("times new roman",17),bg="blue",activebackground='midnight blue',fg="white",activeforeground='white',cursor='hand2')
        self.btn_Clear.place(x=300,y=15,width=200,height=50)
#........................................................................
    def upload_img(self):   
        self.filename = filedialog.askopenfilename(filetypes=(("Jpg File","*.jpg"),("Png File","*.png"),("ALL Files","*.*")))
        print(self.filename)
        self.img=Image.open(self.filename)
        self.img = ImageTk.PhotoImage(self.img)
        self.imglabel.config(image=self.img)
        self.imglabel=self.imglabel
        self.FilePath.set(self.filename)
        
    def prediction(self):
        self.Imgpath=self.FilePath.get()
        data_dir= 'C://Users//vikas//OneDrive//Desktop//signatur//data//'
        model = keras.models.load_model('signature_model.h5')
        batch_size=3
        img_height= 180
        img_width= 180
        train_ds = image_dataset_from_directory(
          data_dir,
          validation_split=0.2,
          subset="training",
          seed=12,
          image_size=(img_height, img_width),
          batch_size=batch_size)
          
        class_names= train_ds.class_names
        
        tesing_img = self.Imgpath
        print(tesing_img)
        
        img = tf.keras.preprocessing.image.load_img(
        tesing_img, target_size=(img_height, img_width)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        self.FilePath.set("")
        score=(100 * np.max(score))
        result=class_names[np.argmax(score)]
        self.FilePath.set("Signature name: "+result+" With Accuracy : "+str(score))
        
    def exit(self):
        self.root.destroy()
        
root=Tk()
obj=Signature_detection(root)
root.mainloop() 

