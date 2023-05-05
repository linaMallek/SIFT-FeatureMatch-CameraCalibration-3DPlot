import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import glob
import math

#les points match soient voisins
#best match
#nombre de point matché 

# Code from:
# https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html

# Prepared for M1IV students, by Prof. Slimane Larabi
#===================================================

print('hello')
img1 = cv2.imread('imag1.jpg',0) # queryImage
img2 = cv2.imread('imag2.jpg',0) # trainImage

# Initiate SIFT detector
sift = cv2.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
print(type(kp2))
#print(des1[0],kp1[0].pt)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)# k choix de distance 
# Sort them in the order of their distance.
print(type(matches))
#matches = sorted(matches, key = lambda x:x.distance)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.65*n.distance:
        #print(m.queryIdx, m.trainIdx, m.distance, n.queryIdx, n.trainIdx, n.distance, n.imgIdx)
        good.append([m])
        
# cv2.drawMatchesKnn expects list of lists as matches.
matched_points1 = np.float32([kp1[m[0].queryIdx].pt for m in good]).reshape(-1, 1, 2)
matched_points2 = np.float32([kp2[m[0].trainIdx].pt for m in good]).reshape(-1, 1, 2)

print(type(matched_points1))
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good, None, flags=2)
plt.imshow(img3),plt.show()

#extract the coordinates (x1,y1) (x2,y2)

good_points = []
for match in good:
    # Get the indices of the corresponding keypoints in kp1 and kp2
    idx1 = match[0].queryIdx
    idx2 = match[0].trainIdx
    # Get the (x, y) coordinates of the keypoints
    x1, y1 = kp1[idx1].pt
    x2, y2 = kp2[idx2].pt
    # Append the coordinates to the list of good points
    good_points.append([(x1, y1), (x2, y2)])

#print (good_points)

#Calibate the camera 

CHECKERBOARD = (9,7)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# max number of iterations=30

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = [] 
error =[]

# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

nx = 7
#Enter the number of inside corners in y
ny = 9
# Extracting path of individual image stored in a given directory
images = glob.glob('images\*.jpg')
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    #ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    """
    If desired number of corner are detected,
    we refine the pixel coordinates and display 
    them on the images of checker board
    """
    if ret == True:
        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
        
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
    
    cv2.imshow('img',img)
    cv2.waitKey(0)

cv2.destroyAllWindows()

h,w = img.shape[:2]

"""
Performing camera calibration by 
passing the value of known 3D points (objpoints)
and corresponding pixel coordinates of the 
detected corners (imgpoints)
"""
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
img1
print("ret: \n")
print(ret)
print("Camera matrix : \n")
print(mtx)

fx=mtx[0][0]
Cx=mtx[0][2]
fy=mtx[1][1]
Cy=mtx[1][2]
b=2
# calculate 3D dimension 
print((good_points[1][1])[1])
l=good_points[1][1]
print(l[1])
D_Coordonnes=[]
for matches in good_points :
     
     x_2D=(b*((matches[1])[0]-Cx)/(matches[1])[0]-(matches[0])[0])
     y_2D=(b*fx*((matches[1])[0]-Cx)/(fy*(matches[1])[0]-(matches[0])[0]))

     Z=8*b/((matches[1])[0]-(matches[0])[0])
     x_3D=(x_2D-Cx)*Z/fx
     y_3D=(x_2D-Cy)*Z/fy
     D_Coordonnes.append((x_3D,y_3D,abs(Z)))

print(D_Coordonnes)



# Extract x, y, and z coordinates from D_Coordonnes
x_coords = [coord[0] for coord in D_Coordonnes]
y_coords = [coord[1] for coord in D_Coordonnes]
z_coords = [coord[2] for coord in D_Coordonnes]

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the points
ax.scatter(x_coords, y_coords, z_coords)

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Scatter Plot')

# Function to handle mouse scroll events
def handle_scroll(event):
    if event.button == 'up':
        ax.set_xlim3d(ax.get_xlim3d()[0] * 0.9, ax.get_xlim3d()[1] * 0.9)
        ax.set_ylim3d(ax.get_ylim3d()[0] * 0.9, ax.get_ylim3d()[1] * 0.9)
        ax.set_zlim3d(ax.get_zlim3d()[0] * 0.9, ax.get_zlim3d()[1] * 0.9)
    elif event.button == 'down':
        ax.set_xlim3d(ax.get_xlim3d()[0] * 1.1, ax.get_xlim3d()[1] * 1.1)
        ax.set_ylim3d(ax.get_ylim3d()[0] * 1.1, ax.get_ylim3d()[1] * 1.1)
        ax.set_zlim3d(ax.get_zlim3d()[0] * 1.1, ax.get_zlim3d()[1] * 1.1)
    fig.canvas.draw()

# Connect the scroll event to the callback function
fig.canvas.mpl_connect('scroll_event', handle_scroll)

# Make the plot interactive
plt.ion()

# Show the plot
plt.show()

# Add a loop to keep the plot open
while True:
    try:
        plt.pause(0.1)
    except KeyboardInterrupt:
        break
