import cv2
import random

def webcam_test():
    # train machine with data of multiple faces
    trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    #capture video from webcam
    webcam = cv2.VideoCapture(0)

    while True:
        #read from webcam
        successful_frame_read, frame = webcam.read()  #retruns is videocaptured successfully and frame

        #grayscale frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # get coordinates of the face
        get_crowd_coordinates = trained_face_data.detectMultiScale(gray_frame)

        #draw rectangle
        for (x, y, w, h) in get_crowd_coordinates:
            cv2.rectangle(frame, (x, y), (x + w, y + h),
                          (random.randrange(255), random.randrange(255), random.randrange(255)), 3)
        #show in window
        cv2.imshow('Clever program', frame)

        #wait 1 sec for key press or it continues
        key = cv2.waitKey(1)

        ## press Q to quit
        if key == 81 or key == 113:
            break

    webcam.release()