import cv2
from random import randrange

#Load Pretrained Data on Face Frontals
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#To Capture Video From Webcam
webcam = cv2.VideoCapture(0) #Argument 0 reads from webcam or we can give the name of a video file

#Iterate Forever Over Frames
while True:

    #Read The Current Frame (first thing is a placeholder)
    successful_frame_read, frame = webcam.read()

    #To Convert The Image To Grayscale i.e Black and White
    grayscaled_img = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)

    #Detect Faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    #Draw Rectangle Around Face (image,point 1, point 2 (by adding point 1 + width and height of rect), color of rect,thickness of line)
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w , y+h), (randrange(256),randrange(256),randrange(256)), 2)

    
    cv2.imshow('Clever Programmer Face Detector',frame)
    key = cv2.waitKey(1)


    #Stop If Key Q Is Pressed (uppercase Q or lowercase q it will quit)
    if key==81 or key==113:
        break

#Release The Videocapture Object
webcam.release()

print(face_coordinates)

