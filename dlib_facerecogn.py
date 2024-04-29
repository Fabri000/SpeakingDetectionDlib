import datetime
import sys
import os
import time
import dlib
import glob
from skimage import io
from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import cv2


def mouth_aspect_ratio(mouth):
	# euclidean distances between vertical mouth landmarks 
    A = dist.euclidean(mouth[2], mouth[3]) # landmarks inner lips - 52,58
	
    # euclidean distance between horizontal mouth landmark 
    C = dist.euclidean(mouth[0], mouth[1]) # landmarks corners of the mouth - 49,55

	# compute the mouth aspect ratio
    mar = A /  C

    return mar 


def blur_image(img,k=50):
    ris = img.copy()
    hh, ww = img.shape[:2]
    hh2 = hh // 2
    ww2 = ww // 2

    # define circles
    axes = [ww2//2,hh2-k]
    yc = hh2
    xc = ww2

    # draw filled circle in white on black background as mask
    mask = np.zeros_like(img)
    mask = cv2.ellipse(mask,(xc,yc),axes,0,0,360,(255,255,255),-1)

    # apply mask to image
    ris = cv2.bitwise_and(img, mask)
    cv2.imshow('frame', frame)
    return ris



# Parameters
MOUTH_AR_THRESH=0.27          # threshold on mouth aspect ratio for detecting the open mouth
EPS=0.27                       # parameter to correct the threshold as the distance increases
K = 50                        # scale factor for the minor axis of the mask
MINIMUM_FACE_AREA = 7000      # parameter to ignore faces that are too close
MAXIMUM_FACE_AREA = 29000     # parameter to ignore faces that are too far away

predictor_path = 'shape_predictor_68_face_landmarks.dat'     # model weights

test = open("test.txt", "w")  # path of the file in which the opening and closing of the mouth is recorded
time0 = 0


# Initializing video capture from the camera
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# Initialization of the model to detect faces
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


while(cap.isOpened()):

    # Timing 
    time.sleep(0.1)

    # Reading and blurring a frame
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = blur_image(frame)
    
    # Detecting faces
    dets = detector(frame, 0)

    for k, d in enumerate(dets):

        # Get landmarks
        shape = predictor(frame, d)
        shape = face_utils.shape_to_np(shape)

        # Calculation of the face area
        area =  d.area()
        
        # Ignore faces that are too close and too far away
        if (area >= MINIMUM_FACE_AREA) & (area <= MAXIMUM_FACE_AREA):

            # Getting landmarks of mouth and of face borders
            mouth = np.array([shape[60],shape[64],shape[62],shape[66]])
            face_borders = [shape[2],shape[14]]
            
            MAX_DISTANCE = frame.shape[0]//2   # width of the blur oval
            distance = dist.euclidean(face_borders[0],face_borders[1]) 
            
            # Calculation of corrected mouth aspect ratio        
            mouth_ar = mouth_aspect_ratio(mouth)
            corrected_mar = mouth_ar + (1 - distance/MAX_DISTANCE)*EPS
            print(f'mount_ar: {mouth_ar}')
            print(f'distance: {distance}')
            print(f'MAX_DISTANCE: {MAX_DISTANCE} , distance/MAX_DISTANCE: {distance/MAX_DISTANCE},  (1 - distance/MAX_DISTANCE): {(1 - distance/MAX_DISTANCE)} f: {(1 - distance/MAX_DISTANCE)*EPS}')
            print(f'correct mar : {corrected_mar}\n')

            # Show contour of the mouth and the correct mar
            mouthHull = cv2.convexHull(mouth)
            cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
            cv2.putText(frame, "MAR: {:.2f}".format(mouth_ar), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7 , (0, 0, 255), 2)
            
            # Check if the mouth is open and stores it on the file
            if corrected_mar > MOUTH_AR_THRESH:
                cv2.putText(frame, "Mouth is Open!", (30,60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
                time0 = time.time()
                test.write(f"OPEN: {time0}\n")
            if time0 != 0 and time.time() - time0 < 2:
                test.write("CLOSED\n")
    
    # Shows the image on the screen
    cv2.imshow('frame', frame)
    
    # To exit and stop execution
    if cv2.waitKey(1) & 0xFF == ord('q'):
        test.close()
        print("q pressed")
        break


cap.release()

cv2.destroyAllWindows()


