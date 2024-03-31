# -*- coding: utf-8 -*-
"""
@author: Deeksha

The following project consists of a trained image recognition model(made using teachable) combined with python GUI(tkinter)to detect
abnormalities in nails to diagnose anaemia or heart disease. This is because there is a change in nail colour in these diseases and 
they are very often ignore(pale in anaemia and blue for heart disease). 
"""

#import necessary models
from tkinter import *
from tkinter import filedialog
from keras.models import load_model 
from PIL import Image, ImageOps, ImageTk 
import numpy as np

#initialising GUI
tk=Tk()
tk.geometry("680x400")
tk.title("Nail diagnosis!")

#get image from user
def get_image_path():
    #tkinter filedialoig used to open the computer's file dialog menu, allowing users to pick an image
    filepath = filedialog.askopenfilename(
        title="Upload Image",
        filetypes=[("Image Files", "*.jpg;*.png;*.jpeg;*.gif")]
    )
    if filepath:
        #Checking if a filepath is received and then using the trained model
        test(filepath)
        
#Trained model(image detection) output
def test(filepath):
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)
    
    # Load the model
    model = load_model("keras_Model.h5", compile=False)
    
    # Load the labels
    class_names = open("labels.txt", "r").readlines()
    
    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    
    # Replace this with the path to your image
    image = Image.open(filepath).convert("RGB")
    
    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    
    # turn the image into a numpy array
    image_array = np.asarray(image)
    
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    
    # Load the image into the array
    data[0] = normalized_image_array
    
    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    
    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", confidence_score)
    
    #Updating labels to display the output(diagnosis)
    if class_name[0]== "0":
        print("anaemic")  
        extraOut.config(text= "You may show signs of anaemia!")
        output.config(text = "Anaemia is largely a nutrional deficiency disorder. Consider an iron-rich diet and supplements. Please seek medical advice if it persists.")
    elif class_name[0] == "1":
        print("heart disease")
        extraOut.config(text = "You may have heart disease...")
        output.config(text = "Your nails may show signs of heart disease. Please visit a doctor for further examination and definitive diagnosis.")
    elif class_name[0]== "2":
        print("healthy")
        extraOut.config(text="You're all good!!")
        output.config(text="Your nails are healthy!! Stay healthy!!")  
    else:
        print(class_name[0])

#Initialising tkinter frame
frm = Frame(tk, width = 300, height = 50,bd=1, bg = "lemon chiffon") 

#Adding the title on the screen
title = Label(tk, text = "Nail it!!", fg = "Midnight blue", font= ("Lexend", 25,"bold"))
title.grid(row = 2, column = 1, padx=10, pady=10)

#Giving a short introduction to the project
intro = Label(tk, text = "Nails can reveal a lot about our health! A simple image of your nails can reveal if you may suffer from anaemia or a heart disease.", font="Sans-Serif, 13", wraplength=700)
intro.grid(row = 3, column= 1, padx=10, pady=10)

#Instructions on how to work the application
start = Label(tk, text ="Add a photo of your nails to get your diagnosis(do not use nail products/polish)", fg="dark slate blue", font=("Lexend", 10,"italic"), wraplength=700)
start.grid(row=4, column=1,padx=10,pady=5)

#Allowing users to upload the image
uploadbtn = Button(tk, text="Upload image of nails: Accepts jpg, jpeg, png and gif files", bg="light slate blue", relief ="ridge", font = "Georgia, 13", command=get_image_path)
uploadbtn.grid(row=5, column=1,padx=10, pady=20)


#Two empty labels used to give diagnosis
extraOut = Label(tk,text=" ", fg = "dark slate blue", wraplength=700, font=("Georgia", 15, "bold"))
extraOut.grid(row=6, column=1, padx=10, pady=5)
output = Label(tk,text = " ", fg="dark slate blue", wraplength=600, font="Georgia,14")
output.grid(row=7,column=1,padx=10,pady=5)

tk.mainloop()
