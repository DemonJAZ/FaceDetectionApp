import cv2

# train machine with data of multiple faces
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Sample immage
img = cv2.imread('Elon_Musk.jpg')

# Convert to grayscale

gray_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detect faces , get coordinates

face_coordinates = trained_face_data.detectMultiScale(gray_img)

print(face_coordinates)

# draw rectangle on image
#cv2.rectangle(img, (135, 127), (135+190, 127+190), (0, 0, 255), 1)
#(x, y, w, h) = face_coordinates
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (120, 0, 255), 3)


cv2.imshow('Clever program', img)
cv2.waitKey()

print('Code Complete\n')