import tkinter as tk
import cv2
from PIL import Image, ImageTk
import random
import pygame
from tensorflow.keras.models import load_model
import numpy as np
import logging
logging.getLogger('absl').setLevel(logging.ERROR)

def detect_sleep(img):
    # Function to detect sleep using the model
    # Returns a boolean value if sleeping or not
    loaded_model = load_model("final_model.h5")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])

            (startX, startY, endX, endY) = box.astype("int")
            cropped_face = img[startY:endY, startX:endX]
                
            cropped_face = cv2.resize(cropped_face, (500, 500))
            
            pred = loaded_model.predict(np.array([cropped_face,]))
            return pred > 0.95


def play_beep():
    pygame.mixer.init()
    pygame.mixer.music.load("beep.mp3")
    pygame.mixer.music.play()


def update():
    ret, frame = vid.read()  # ret is the return value if success, frame is the image object we want
    if ret:
        # Detect sleep using the model. detect_sleep should return a boolean
        is_sleeping = detect_sleep(frame)
        if is_sleeping:  # if sleeping, display a "WAKE UP" text in the tkinter UI
            play_beep()
            label_sleep.config(text="WAKE UP", fg="red", font=("Arial", 24, "bold"))
        else:  # else, nothing will be displayed
            label_sleep.config(text="", fg="black")

        # Update tkinter canvas with the live camera feed
        photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        canvas.photo = photo  # Keeping reference to avoid garbage collection

    # Call update function again after 100 milliseconds. decrease this to increase fps.
    window.after(100, update)


# Create a Tkinter window
window = tk.Tk()
window.title("Sleep Detection")

# Open video source, 0 for primary camera
vid = cv2.VideoCapture(0)

# Create a Tkinter canvas to display the video feed
canvas = tk.Canvas(window, width=vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
canvas.pack()

# Tkinter Label to display the "WAKE UP" message
label_sleep = tk.Label(window, text="", font=("Arial", 24, "bold"))
label_sleep.pack()

# Start camera feed and model
update()

window.mainloop()
