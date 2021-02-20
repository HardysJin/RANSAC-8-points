'''
Install opencv:
pip install opencv-python==3.4.2.16
pip install opencv-contrib-python==3.4.2.16
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--UseRANSAC", type=int, default=0 )
# parser.add_argument("--image1", type=str,  default='data/myleft.jpg' )
# parser.add_argument("--image2", type=str,  default='data/myright.jpg' )
# parser.add_argument("--image1", type=str,  default='data/921919841_a30df938f2_o.jpg' )
# parser.add_argument("--image2", type=str,  default='data/4191453057_c86028ce1f_o.jpg' )
parser.add_argument("--image1", type=str,  default='data/7433804322_06c5620f13_o.jpg' )
parser.add_argument("--image2", type=str,  default='data/9193029855_2c85a50e91_o.jpg' )
args = parser.parse_args()

print(args)

def normalize(pts):
    # The points are translated so that their centroid is at the origin.
    # 1. Find the centroid of the pts (find the mean x and mean y value)
    mean_x = np.mean(pts[:, 0])
    mean_y = np.mean(pts[:, 1])

    # 2. Compute the mean distance of all the pts from this centroid
    distances = np.sqrt((pts[:, 0] - mean_x) ** 2 + (pts[:, 1] - mean_y) ** 2)
    mean_dist = np.mean(distances)
    if mean_dist == 0:
        print(pts)

    # The points are then scaled isotropically so that the average distance from the origin is equal to sqrt(2).
    # 3. Construct a 3 by 3 matrix that would translate the points so that the mean distance would be sqrt(2)
    s = np.sqrt(2) / mean_dist
    # transformations matrix
    T = np.array([[s, 0, -mean_x * s],
                  [0, s, -mean_y * s],
                  [0, 0,     1      ]])

    # add additional column of 1 to pts => [x, y, 1]
    N = len(pts)
    norm = np.c_[ pts, np.ones(N) ].T
    # normalize
    norm = np.dot(T, norm).T

    return norm, T
    

def FM_by_normalized_8_point(pts1,  pts2):
    # F, _ = cv2.findFundamentalMat(pts1,pts2,  cv2.FM_8POINT )
    # print(F)
    # comment out the above line of code. 

    # Your task is to implement the algorithm by yourself.
    # Do NOT copy&paste any online implementation. 

    # workflow from http://www.cs.cmu.edu/~16385/s17/Slides/12.4_8Point_Algorithm.pdf, page. 18
    # 0. (Normalize points)
    norm_pts1, T1 = normalize(pts1)
    norm_pts2, T2 = normalize(pts2)
    # print(norm_pts1[0])

    # 1. Construct the N x 9 matrix A
    N = len(pts1)
    A = np.zeros([N, 9])
    for i in range(N):
        [x1, y1, _] = norm_pts1[i]
        [x2, y2, _] = norm_pts2[i]
        # 𝑥𝑥′ 𝑥𝑦′ 𝑥 𝑦𝑥′ 𝑦𝑦′ 𝑦 𝑥′ 𝑦′ 1
        A[i] = [x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1]

    # 2. Find the SVD of ATA
    ATA = np.dot(A.T, A)
    # print(ATA.shape)
    u, s, vT = np.linalg.svd(ATA)

    # 3. Entries of F are the elements of column of V corresponding to the least singular value
    F = vT.T[:,-1].reshape([3,3]).T
    # print(F)

    # 4. Enforce rank 2 constraint on F -> make sigma_3 = 0
    # SVD on F to find the sigma
    u, s, vT = np.linalg.svd(F)
    # force the last sigma zero
    s[2] = 0
    # reconstruct s to diag matrix
    s = np.diag(s)
    # print(s)
    # reconstruct F by rank 2 sigma
    F = np.dot(np.dot(u, s), vT)
    # print(F)

    # 5. (Un-normalize F) -> F = (T')^T F_bar T
    F = np.dot(np.dot(T2.T, F), T1)
    F = F/F[2,2]
    # print(F)
	
    # F:  fundmental matrix
    return  F

def is_unique(X):
    if type(X) == np.ndarray:
        X = X.tolist()
    uniq = []
    for x in X:
        if x not in uniq:
            uniq.append(x)
        else:
            return False
    return True

def FM_by_RANSAC(pts1,  pts2):
    # F, mask = cv2.findFundamentalMat(pts1,pts2,  cv2.FM_RANSAC )
    # print(F)
    # comment out the above line of code. 
	
    # Your task is to implement the algorithm by yourself.
    # Do NOT copy&paste any online implementation. 

    n = 0
    threshold=3
    confidence=0.99

    # assume 50% inlier
    M = int(np.log(1 - confidence) / np.log(1 - 0.5 ** 8))
    # print(M)

    N = len(pts1)

    _pts1 = np.c_[pts1, np.ones(N)]
    _pts2 = np.c_[pts2, np.ones(N)]

    for i in range(M):
        # find 8 pairs of unique corresponding points 
        idx = np.random.choice(np.arange(len(pts1)), 8, replace = False)
        _8pt_1 = pts1[idx,:]
        _8pt_2 = pts2[idx,:]
        # compute F based on random 8 pairs points
        F_i = FM_by_normalized_8_point(_8pt_1, _8pt_2)

        inlier_mask = np.zeros(N)
        for j in range(N):
            # x'*F*x < 0.001
            # distance from point to line: d^2 = |ax+by+c|^2/(a^2+b^2)
            # https://github.com/kokerf/vision_kit/blob/master/module/epipolar_geometry/src/fundamental.cpp
            # Given F, compute error (distance from epiline) for L1 and L2
            vec = np.dot(_pts2[j], F_i)
            [x,y] = vec[0:2] 
            e1 = (np.dot(vec, _pts1[j]) ** 2) / (x**2 + y**2)

            vec = np.dot(F_i, _pts1[j])
            [x,y] = vec[0:2] 
            e2 = (np.dot(vec, _pts2[j]) ** 2) / (x**2 + y**2)

            # choose max of two errors
            e = max(e1, e2)

            # if error less than threshold, this point is inlier
            if e < threshold:
                inlier_mask[j] = 1

        
        n_i = sum(inlier_mask)
        # if n_i < 8, ignore this iteration because FM_by_normalized_8_point reuqires at least 8 points
        if n_i < 8: continue
        # if the number of inliers is higher, it is better fit
        if n_i > n:
            n = n_i
            print("Num inliers: ", n)
            mask = inlier_mask.reshape(N, 1)
            # recompute the F based on new set of inliers
            tmp_pts1 = pts1[mask.ravel() == 1]
            tmp_pts2 = pts2[mask.ravel() == 1]

            F = FM_by_normalized_8_point(tmp_pts1, tmp_pts2)
	
    # F:  fundmental matrix
    # mask:   whetheter the points are inliers
    return  F, mask

	
img1 = cv2.imread(args.image1,0) 
img2 = cv2.imread(args.image2,0)  

sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

good = []
pts1 = []
pts2 = []

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)
		
		
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

# img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None)
# plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
# plt.show()

F = None
if args.UseRANSAC:
    # set comparison
    F1, mask1 = cv2.findFundamentalMat(pts1,pts2,  cv2.FM_RANSAC )
    _pts1 = pts1[mask1.ravel()==1]
    _pts2 = pts2[mask1.ravel()==1]

    F,  mask = FM_by_RANSAC(pts1,  pts2)
    # We select only inlier points
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]
else:
    F = FM_by_normalized_8_point(pts1,  pts2)
	

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2
	

# Find epilines corresponding to points in second image,  and draw the lines on first image
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,  F)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)

if args.UseRANSAC:
    lines1 = cv2.computeCorrespondEpilines(_pts2.reshape(-1,1,2), 2,  F1)
    lines1 = lines1.reshape(-1,3)
    img7,img8 = drawlines(img1,img2,lines1,_pts1,_pts2)

    fig = plt.figure()
    fig.suptitle("Comparison between My Implementation and OpenCV")

    plt.subplot(221),plt.imshow(img5), plt.title("My Implementation")
    plt.subplot(222),plt.imshow(img6)
    plt.subplot(223),plt.imshow(img7), plt.title("OpenCV")
    plt.subplot(224),plt.imshow(img8)
    plt.show()
else: 
    img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
    plt.subplot(121),plt.imshow(img5)
    plt.subplot(122),plt.imshow(img6)
    plt.show()

# Find epilines corresponding to points in first image, and draw the lines on second image
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)

if args.UseRANSAC:
    lines2 = cv2.computeCorrespondEpilines(_pts1.reshape(-1,1,2), 1,F1)
    lines2 = lines2.reshape(-1,3)
    img7,img8 = drawlines(img2,img1,lines2,_pts2,_pts1)

    fig = plt.figure()
    fig.suptitle("Comparison between My Implementation and OpenCV")

    plt.subplot(221),plt.imshow(img4)
    plt.subplot(222),plt.imshow(img3), plt.title("My Implementation")
    plt.subplot(223),plt.imshow(img8)
    plt.subplot(224),plt.imshow(img7), plt.title("OpenCV")

    plt.show()
else:
    img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
    plt.subplot(121),plt.imshow(img4)
    plt.subplot(122),plt.imshow(img3)
    plt.show()