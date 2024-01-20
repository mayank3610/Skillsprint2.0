import cv2
import os
import  time

haar_file="haarcascade_frontalface_default.xml"

datasets="C:/Users/Preet/PycharmProjects/facedetect/dataset"

name="Mayank"



path = os.path.join(datasets, name)
if not os.path.isdir(path):
    os.mkdir(path)

#size of image
(width, height) = (640, 480)
#cascade classifier is use for object detection
face_cascade = cv2.CascadeClassifier(haar_file)
cap = cv2.VideoCapture(0)
# if camera is on then it will show true
print("Webcam is open? ", cap.isOpened())
# wait for the camera to turn on (just to be safe, in case the camera needs time to load up)
time.sleep(2)
#Takes pictures of detected face and saves them
count = 1
print("Taking pictures...")
# this takes 100 pictures of your face. Change this number if you want.
# Having too many images, however, might slow down the program
while count < 101:
    # im = camera stream
    ret, frame= cap.read()
    # if it recieves something from the webcam...
    if ret == True:
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detect face using the haar cascade file
        faces = face_cascade.detectMultiScale(img_gray, 1.3, 4)  #(source,scale,min_neighbours)
        for (x,y,w,h) in faces:
            # draws a rectangle around your face when taking pictures
            # this is to create a ROI (region of interest) so it only takes pictures of your face
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            # define 'face' as the inside of the rectangle we made above and make it grayscale
            face = img_gray[y:y + h, x:x + w]
            # resize the face images to the size of the 'face'
            face_resize = cv2.resize(face, (width, height))
            # it can save all the images with number
            cv2.imwrite('%s/%s.png' % (path,count), face_resize)
        count += 1
        cv2.imshow('Face Capturing', frame)
        if cv2.waitKey(1) == ord("q"):
            break
print("Your face has been created.")
cap.release()
cv2.destroyAllWindows()
