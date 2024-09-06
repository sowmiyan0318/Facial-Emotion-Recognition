import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace
import numpy as np
from keras.preprocessing import image

# Load the Haar cascade classifier XML file for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Start capturing video from the default webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture
    ret, test_img = cap.read()

    # Convert the frame to grayscale
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces_detected = face_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x, y, w, h) in faces_detected:
        # Draw a rectangle around the detected face
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)

        # Crop the region of interest (face area) from the image
        roi_gray = gray_img[y:y + h, x:x + w]

        # Resize the cropped face image to 48x48
        roi_gray = cv2.resize(roi_gray, (48, 48))

        # Convert the grayscale image to a 3-channel image
        roi_gray = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)

        # Make predictions for the emotions using the pre-trained DeepFace model
        predictions = DeepFace.analyze(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB), actions=['emotion'], enforce_detection=False)

        # Get the dominant emotion from the predictions
        if predictions:
            predicted_emotion = predictions[0]['dominant_emotion']
        else:
            predicted_emotion = 'Unknown'

        # Display the predicted emotion on the frame
        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the resulting frame with the labeled emotion
    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis ', resized_img)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(10) == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()