import sys
import os
import dlib
import glob
from skimage import io
from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import cv2
import time 

def mouth_aspect_ratio(mouth):
	# compute the euclidean distances between the two sets of
	# vertical mouth landmarks (x, y)-coordinates
    A = dist.euclidean(mouth[2], mouth[3]) # 52, 58
	# compute the euclidean distance between the horizontal
	# mouth landmark (x, y)-coordinates
    C = dist.euclidean(mouth[0], mouth[1]) # 49, 55

	# compute the mouth aspect ratio
    mar = A /  C

	# return the mouth aspect ratio 
    return mar 



def blur_image(img):
    ris = img.copy()
    hh, ww = img.shape[:2]
    hh2 = hh // 2
    ww2 = ww // 2

    # define circles
    axes = [ww2//2,hh2-K]
    yc = hh2
    xc = ww2

    # draw filled circle in white on black background as mask
    mask = np.zeros_like(img)
    mask = cv2.ellipse(mask,(xc,yc),axes,0,0,360,(255,255,255),-1)


    # apply mask to image
    ris = cv2.bitwise_and(img, mask)
    cv2.imshow('frame', frame)
    return ris


cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')

#Parameterss
MOUTH_AR_THRESH=0.65
K = 50 # scale factor for the minor axis of the mask
MINIMUM_FACE_AREA = 7000 # more distant the less
MAXIMUM_FACE_AREA = 29000 # closer more
EPS=0.30

predictor_path = 'shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

#index of the facial landmark for the mouth
(mStart,mEnd)=(48,68)


while(cap.isOpened()):
    time.sleep(0.1)
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = blur_image(frame)
    
    dets = detector(frame, 0)
    for k, d in enumerate(dets):
        
        shape = predictor(frame, d)
        shape = face_utils.shape_to_np(shape)
        
        
        MAX_DISTANCE = frame.shape[0]//2 - K
        print(f'face detected {dets}')
        area =  d.area()
        print(f'face area: {area}')

        
        # verifica se la bocca è aperta
        if (area >=MINIMUM_FACE_AREA) & (area <= MAXIMUM_FACE_AREA):
            
            mouth = np.array([shape[60],shape[64],shape[62],shape[66]])
            mouth_ar = mouth_aspect_ratio(mouth)

            face_borders = [shape[2],shape[14]]
            distance = dist.euclidean(face_borders[0],face_borders[1])

            corrected_mar = mouth_ar + (1 - distance/MAX_DISTANCE)*EPS

            print(f'mount_ar: {mouth_ar}')
            print(f'distance: {distance}')
            print(f'MAX_DISTANCE: {MAX_DISTANCE} , distance/MAX_DISTANCE: {distance/MAX_DISTANCE},  (1 - distance/MAX_DISTANCE): {(1 - distance/MAX_DISTANCE)} f: {(1 - distance/MAX_DISTANCE)*EPS}')
            print(f'correct mar : {corrected_mar}')

            mouthHull = cv2.convexHull(mouth)
            cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
            cv2.putText(frame, "CORRECTED MAR: {:.2f}".format(corrected_mar), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if corrected_mar > MOUTH_AR_THRESH:
                cv2.putText(frame, "Mouth is Open!", (30,60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
            
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("q pressed")
        break


cap.release()

cv2.destroyAllWindows()


