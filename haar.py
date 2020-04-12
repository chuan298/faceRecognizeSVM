import cv2
import time
import pickle
import numpy as np
from imutils.video import FPS 
from imutils.video import VideoStream
# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read())
le = pickle.loads(open("output/le.pickle", "rb").read())
# To capture video from webcam. 
cap = cv2.VideoCapture(0)
# To use a video file as input 
# cap = cv2.VideoCapture('filename.mp4')

while True:
    # Read the frame
    startTime = time.time()
    _, img = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.5, 4)

    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        if w < 100 or h < 100:
            continue
        face = img[y:y+h, x:x+w]
        faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
            (96, 96), (0, 0, 0), swapRB=True, crop=False) 
        embedder.setInput(faceBlob)
        vec = embedder.forward()

		# perform classification to recognize the face
        preds = recognizer.predict_proba(vec)[0]
        j = np.argmax(preds)
        proba = preds[j]
        name = le.classes_[j]

        text = "{}: {:.2f}%".format(name, proba * 100)
        yname = y - 10 if y - 10 > 10 else y + 10
        cv2.putText(img, text, (x, yname),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
 
    # Display
    cv2.imshow('img', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(1) & 0xff
    if k==27:
        break
    endTime = time.time()
    t = endTime - startTime
    print("time run :" + str(t))
# Release the VideoCapture object
cap.release()

