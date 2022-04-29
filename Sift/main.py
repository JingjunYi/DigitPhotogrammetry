import os
import numpy as np
import cv2
import sift
from matplotlib import pyplot as plt
from time import time


def sift_match(img1_pth, img2_pth, output_dir, sigma=1.6, interval_num=3, blur=0.5, border_width=5):
    time0 = time()
    img1 = cv2.imread(img1_pth) 
    img2 = cv2.imread(img2_pth)
    img1_match1 = img1.copy()
    img2_match1 = img2.copy()
    img1_match2 = img1.copy()
    img2_match2 = img2.copy()
    img1_match3 = img1.copy()
    img2_match3 = img2.copy()
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    ## Compute SIFT keypoints and descriptors
    time1 = time()
    print('Feature detecting.')
    mysift = sift.Sift(sigma, interval_num, blur, border_width)
    mysift.detect_keypoints(img1_gray)
    keypoint1 = mysift.keypoints
    desc1 = mysift.gen_desc()
    np.savetxt(os.path.join(output_dir, 'desc1.txt'), desc1, fmt='%.2f')
    mysift.detect_keypoints(img2_gray)
    keypoint2 = mysift.keypoints
    desc2 = mysift.gen_desc()
    np.savetxt(os.path.join(output_dir, 'desc2.txt'), desc1, fmt='%.2f')
    time2 = time()
    print('Feature extracted in {}ms.'.format((time2 - time1) * 1000))
    
    ## Visualization SIFT feature point
    cv2.drawKeypoints(img1, keypoint1, img1, (0,255,255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #cv2.imshow('feature1', img1)
    #cv2.waitKey()
    cv2.imwrite(os.path.join(output_dir, 'feature1.png'), img1)
    cv2.drawKeypoints(img2, keypoint2, img2, (0,255,255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #cv2.imshow('feature22', img2)
    #cv2.waitKey()
    cv2.imwrite(os.path.join(output_dir, 'feature2.png'), img2)
    #cv2.destroyAllWindows()
    
    ## FLANN match
    time3 = time()
    print('Feature matching.')
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)
    matchesMask1 = [[0,0] for i in range(len(matches))]
    matchesMask2 = [[0,0] for i in range(len(matches))]
    matchesMask3 = [[0,0] for i in range(len(matches))]
    threshold1 = 0.5
    threshold2 = 0.7
    threshold3 = 0.9
    for i, (m, n) in enumerate(matches):
        if m.distance < threshold1 * n.distance:
            matchesMask1[i]=[1,0]
        if m.distance < threshold2 * n.distance:
            matchesMask2[i]=[1,0]
        if m.distance < threshold3 * n.distance:
            matchesMask3[i]=[1,0]
    time4 = time()
    print('Feature matched in {}ms.'.format((time4 - time3) * 1000))
    
    ## Visualization SIFT match
    draw_params1 = dict(matchColor = (0,255,0), singlePointColor=(0,255,255), matchesMask=matchesMask1, flags=0)
    draw_params2 = dict(matchColor = (0,255,0), singlePointColor=(0,255,255), matchesMask=matchesMask2, flags=0)
    draw_params3 = dict(matchColor = (0,255,0), singlePointColor=(0,255,255), matchesMask=matchesMask3, flags=0)
    match_img1 = cv2.drawMatchesKnn(img1_match1, keypoint1, img2_match1, keypoint2, matches, None, **draw_params1)
    match_img2 = cv2.drawMatchesKnn(img1_match2, keypoint1, img2_match2, keypoint2, matches, None, **draw_params2)
    match_img3 = cv2.drawMatchesKnn(img1_match3, keypoint1, img2_match3, keypoint2, matches, None, **draw_params3)
    #cv2.imshow(os.path.join(output_dir, 'match1.png'), match_img1)
    #cv2.waitKey()
    cv2.imwrite(os.path.join(output_dir, 'match1.png'), match_img1)
    cv2.imwrite(os.path.join(output_dir, 'match2.png'), match_img2)
    cv2.imwrite(os.path.join(output_dir, 'match3.png'), match_img3)
    #cv2.destroyAllWindows()
    time5 = time()
    print('Process finished in {}ms.'.format((time5 - time0) * 1000))


if __name__ == '__main__':

    output_dir1 = 'pair1'
    output_dir2 = 'pair2'
    img11_pth = '../11.png' 
    img12_pth = '../12.png'
    img21_pth = '../21.png' 
    img22_pth = '../22.png'
    sift_match(img11_pth, img12_pth, output_dir1)
    sift_match(img21_pth, img22_pth, output_dir2)