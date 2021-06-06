import cv2

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('Elon_Musk.jpg')



cv2.imshow('Clever program',img)
cv2.waitKey()

print('Code Complete')