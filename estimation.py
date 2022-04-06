# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 22:29:16 2021

@author: pooja
"""
import numpy as np
#import matplotlib
import matplotlib.pyplot as plt
import imutils
from numpy.linalg import inv
from numpy.linalg import svd
from random import sample
import cv2

def main():
    x= []
    y= []
    
    cap = cv2.VideoCapture('data\Ball_travel_10fps.mp4')
    #cap = cv2.VideoCapture('data\Ball_travel_2_updated.mp4')
    
    # Check if video opened successfully
    if (cap.isOpened()== False):
      print("Error opening video stream or file")
      
    # Read until video is completed
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if ret == True:
            
            #resizing video dimensions
            frame = imutils.resize(frame, width=500)
            frame_ht = frame.shape[0]
           
            #resized video display
            cv2.imshow('Frame', frame)
            
            #mask = 1 at all red/non-white pixels 
            mask = np.all(frame != [255, 255, 255], axis= -1)
            #r is an array of y co-ordinates, c is an array of x co-ordinates wherever mask returns 1
            r,c= mask.nonzero()
    
            #vertical center of red blob
            h_max = np.amax(r)
            h_min = np.amin(r)
            h_cen = (h_max + h_min)/2
            
            #horizontal center of red blob
            d_max = np.amax(c)
            d_min = np.amin(c)
            d_cen = (d_max + d_min)/2
            
            #creating dataset: appending center of ball for each frame
            x.append(d_cen)
            y.append(frame_ht - h_cen)
            
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break 
     
        # Break the loop
        else:
            break
     
    
    # When everything done, release the video capture object
    cap.release()
    
    
    #plot dataset
    plt.plot(x,y,'y*',label='co-ordinates')
    plt.legend()
    
    #SLS
    yest_sls= sls(x,y)
    plt.plot(x,yest_sls[0],'b',label='SLS')
    plt.legend()
    
    #TLS
    yest_tls= tls(x,y)
    plt.plot(x,yest_tls[0],'orange',label='TLS')
    plt.legend()
    
    #RANSAC
    # 3 samples, 0.999 probability that we get a good sample , 25% outlier ratio e, threshhold is 10
    yest_ransac= ransac(x,y,3,0.999,0.25,10)
    plt.plot(x,yest_ransac,'r',label='RANSAC')
    plt.plot(x,yest_ransac+10,'pink')
    plt.plot(x,yest_ransac-10,'pink')
    plt.legend()
    
    
    # Closes all the frames
    cv2.destroyAllWindows()

def sls(x1,y1):
    #x^1
    x1 = np.array(x1)
    n = x1.shape[0]
    x1 = x1.reshape(n,1)
    
    #x^0
    x0 = np.ones(n)
    x0 = x0.reshape(n,1)
    
    #x^2
    x2 = np.square(x1)
    
    #forming the a matrix 
    a = np.hstack((x0,x1,x2))
    
    #solving for curve parameters that minimize standard least squares
    b = inv(a.T.dot(a)).dot(a.T).dot(y1)
    
    #sls estimates for given data
    y_tilde = np.array([b[2]*v**2 + b[1]*v + b[0] for v in x1])
    
    return y_tilde,b

def tls(x1,y1):
    #x^1
    x1 = np.array(x1)
    n = x1.shape[0]
    x1 = x1.reshape(n,1)
    
    #x^0
    x0 = np.ones(n)
    x0 = x0.reshape(n,1)
    
    #x^2
    x2 = np.square(x1)
    
    #forming the a matrix 
    a = np.hstack((x0,x1,x2))
    
    y1 = np.array(y1).reshape(n,1)
    
    #[a y]
    ay= np.column_stack((a,y1))
    
    I = np.identity(a.shape[1])
    
    #minimum singular value of [a y]
    sigma = np.min(svd(ay)[1])

    #solving for curve parameters that minimize total least squares
    b = inv(a.T.dot(a)- sigma**2*(I)).dot(a.T).dot(y1)
    
    #tls estimates for given data
    y_tilde = np.array([b[2]*v**2 + b[1]*v + b[0] for v in x1])
    
    return y_tilde,b
     
# s= samples, p= desired probability that we get a good sample, e= outlier ratio e, th= threshhold 
def ransac(x1,y1,s,p,e,th):
    #converting  to a list of tuples
    data = list(zip(x1, y1))
    
    #optimum no of trials
    t = int(np.log(1-p)/np.log(1-(1-e)**s))
    
    #store curve parameter for every iteration
    all_b = []
    #store inliers parameter for every iteration
    inliers = []
    
    for iter in range(0,t):
        #inliers count for this trial
        inliers_iter = 0
        #sampling 3 datapoints
        sample_iter= sample(data,3)
        
        
        #unzipping tuples into arrays
        x_iter= np.array(list(zip(*sample_iter))[0])
        y_iter= np.array(list(zip(*sample_iter))[1])
        
        #estimating curve parameters using sls
        yest_iter,b_iter= sls(x_iter,y_iter)
        print(b_iter)
        #count datapoints within curve threshhold
        for point in data:
            sls_iter = b_iter[2]*point[0]**2 + b_iter[1]*point[0] + b_iter[0]
            if point[1]< sls_iter+th and point[1]> sls_iter-th:
                inliers_iter = inliers_iter + 1
        
        #append count of inliers and curve parameters for the trial
        inliers.append(inliers_iter)
        all_b.append(b_iter)
    
    #extracting curve parameters for max inliers
    b_optimum = all_b[np.argmax(inliers)] 
    
    #ransac estimates for given data
    ransac = np.array([b_optimum[2]*v**2 + b_optimum[1]*v + b_optimum[0] for v in x1])
    return ransac
    
    

if __name__ == "__main__":
    main()