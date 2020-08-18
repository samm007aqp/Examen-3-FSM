#https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import yaml
import math
import plot3D 
img1 = cv.imread('house2.jpg',0)  #queryimage # left image
img2 = cv.imread('house1.jpg',0) #trainimage # right image


def write_matrix_to_textfile(a_matrix, file_to_write):
	def compile_row_string(a_row):
		return str(a_row).strip(']').strip('[')

	with open(file_to_write, 'w') as f:
		for row in a_matrix:
			for x in row:
				f.write(str(x)+',')
			f.write('\n')
	return True

#cv.imshow('sd',img1)
#cv.waitKey(0)
#cv.destroyAllWindows()

sift = cv.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)


#Brute force matcher
#bf = cv.BFMatcher(cv.NORM_L1, crossCheck = False)
#bmatches = bf.match(des1, des2)
#bmatches = sorted(bmatches, key = lambda x : x.distance)


# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)
good = []
pts1 = []
pts2 = []
print( len(matches) )

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
	if m.distance < 0.80*n.distance:
		#print(m.distance)
		good.append([m])
		pts2.append(kp2[m.trainIdx].pt)
		pts1.append(kp1[m.queryIdx].pt)

print(len(good))
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)


F, mask = cv.findFundamentalMat(pts1,pts2,cv.LMEDS)
# We select only inlier points
pts1 = pts1[mask.ravel()==1] ## just the values with 1 in the mask
#print( len(pts1) )  91 
pts2 = pts2[mask.ravel()==1]
#print(F)
#####################################################################################
#####################################################################################

#intrinsics = np.load('intrinsics.npz')
#K1 = intrinsics['K1']
#K2 = intrinsics['K2']
#print(K1)# segun lo imprimido las 2 matrices son iguales
#print(K2)

K_l_ =[]
dist_coef =[]
with open(r'calibration_matrix.yaml') as file:
	camera_data = yaml.full_load(file)
	cam_matrix, dist_coef = camera_data.items()
	K_l = np.array(cam_matrix[1],np.float32)
	#print(K_l)
	#print(dist_coef)
ptsl = cv.undistortPoints(np.expand_dims(np.float32(pts1), axis=1), cameraMatrix=K_l, distCoeffs=None)
ptsr = cv.undistortPoints(np.expand_dims(np.float32(pts2), axis=1), cameraMatrix=K_l, distCoeffs=None)
E = K_l.T.dot(F).dot(K_l)
#print(pts1.T.shape) # (2,91)
#print(np.expand_dims(pts1, axis=1).shape) (91,1,2)
U,S,V = np.linalg.svd(E)
diag_110 = [ [1 ,0, 0], [0 ,1, 0] ,[0, 0 ,0]]
#newE1 = U.dot(diag_110).dot(V.T)
newE = U.dot(diag_110).dot(np.transpose(V))
U,S,V = np.linalg.svd(newE)
W = np.array([[0 ,-1, 0], [1 ,0, 0] ,[0, 0 ,1]])
r1 = U.dot(W).dot(np.transpose(V))
r2 = U.dot(np.transpose(W)).dot(np.transpose(V));

R1, R2, T = cv.decomposeEssentialMat(E)
points, R, t, mask = cv.recoverPose(E, ptsl, ptsr)
#T and t are equal
M_r = np.hstack((R, t))
M_l = np.hstack((np.eye(3, 3), np.zeros((3, 1))))

P1 = K_l.dot(M_l) 
P2 = K_l.dot(M_r)

pts1 = np.array(pts1,np.float32)
pts2 = np.array(pts2,np.float32)

point_4d_hom = cv.triangulatePoints(P1,P2, np.expand_dims(pts1, axis=1) , np.expand_dims(pts2, axis=1) )

point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
point_3d = point_4d[:3, :].T

point_3d = np.round(point_3d,2)
#with open('outfile.txt','w') as f:
#    for line in point_3d:
#        np.savetxt(f, line, fmt='%.2f')
write_matrix_to_textfile(point_3d,"salida_puntos.txt")


print(point_3d.shape)
plot3D.Draw3D(point_3d)




