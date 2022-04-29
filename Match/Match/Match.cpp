// Match.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <cmath>
#include "FeatureExtraction.h"
#include "FeatureMatch.h"

using namespace std;
using namespace cv;


int main()
{
    cv::utils::logging::setLogLevel(utils::logging::LOG_LEVEL_SILENT);
    string mode;
    cout << "SIFT or MORAVEC ?" << endl;
    cin >> mode;
    
    string img11_pth = "../../11.png";
    string img12_pth = "../../12.png";
    int dist_x = 81;
    int dist_y = 24;
    string img21_pth = "../../21.png";
    string img22_pth = "../../22.png";


    if (mode == "SIFT")
    {
        ////SIFT Corelation Coefficient Match
        vector<Point3f> siftfeature_l;
        vector<Point3f> siftfeature_r;
        vector<Point3f> sfit_matched_point_l;
        vector<Point3f> sfit_matched_point_r;
        //match1->match feature points of two images
        //FeatureExtraction::siftfeature_extraction(img21_pth,  500, siftfeature_l);
        //FeatureExtraction::siftfeature_extraction(img22_pth,  500, siftfeature_r);
        //FeatureMatch::correlationcoefficient_match1(img11_pth, img12_pth, siftfeature_l, siftfeature_r, 9, 0.9);
        //match2->match based on feature points of left image
        FeatureExtraction::siftfeature_extraction(img11_pth, 50, siftfeature_l);
        cout << "SIFT Feature extracted!!!" << endl;
        FeatureMatch::correlationcoefficient_match2(img11_pth, img12_pth, dist_x, dist_y, siftfeature_l, 9, 15, 0.9, sfit_matched_point_l, sfit_matched_point_r);
        cout << "NCC matching done!!!" << endl;

        ////Least Square Method Match
        FeatureMatch::leastsquare_match(img11_pth, img12_pth, sfit_matched_point_l, sfit_matched_point_r, 9);
        cout << "Least square matching done!!!" << endl;

        destroyAllWindows();


        ////SIFT Descriptor Match
        /*int sift_feature_num = 500;
        int final_feature_num = 30;
        Mat img_l = imread(img21_pth);
        Mat img_r = imread(img22_pth);
        vector<KeyPoint> keypoint1, keypoint2;
        Mat desc_l, desc_r;
        Ptr<Feature2D> siftfeature = SIFT::create(sift_feature_num);
        siftfeature->detect(img_l, keypoint1);
        siftfeature->detect(img_r, keypoint2);
        siftfeature->compute(img_l, keypoint1, desc_l);
        siftfeature->compute(img_r, keypoint2, desc_r);
        BFMatcher matcher;
        vector<DMatch> matches;
        matcher.match(desc_l, desc_r, matches);
        nth_element(matches.begin(), matches.begin() + final_feature_num, matches.end());
        matches.erase(matches.begin() + final_feature_num, matches.end());
        Mat img_match;
        drawMatches(img_l, keypoint1, img_r, keypoint2, matches, img_match);
        imshow("Match Result", img_match);
        waitKey(0);
        imwrite("../siftmatch.png", img_match);*/
    }

    else if (mode=="MORAVEC")
    {
        ////Moravec Corelation Coefficient Match
        vector<Point3f> moravecfeature_l;
        vector<Point3f> moravecfeature_r;
        vector<Point3f> moravec_matched_point_l;
        vector<Point3f> moravec_matched_point_r;
        //match1->match feature points of two images
        //FeatureExtraction::moravecfeature_extraction(img11_pth, 5, 9, 200, moravecfeature_l);
        //FeatureExtraction::moravecfeature_extraction(img12_pth, 5, 9, 200, moravecfeature_r);
        //FeatureMatch::correlationcoefficient_match1(img11_pth, img12_pth, moravecfeature_l, moravecfeature_r, 9, 0.7);
        //match2->match based on feature points of left image
        FeatureExtraction::moravecfeature_extraction(img11_pth, 5, 9, 2000, moravecfeature_l);
        cout << "MORAVEC Feature extracted!!!" << endl;
        FeatureMatch::correlationcoefficient_match2(img11_pth, img12_pth, dist_x, dist_y, moravecfeature_l, 9, 15, 0.9, moravec_matched_point_l, moravec_matched_point_r);
        cout << "NCC matching done!!!" << endl;

        ////Least Square Method Match
        FeatureMatch::leastsquare_match(img11_pth, img12_pth, moravec_matched_point_l, moravec_matched_point_r, 9);
        cout << "Least square matching done!!!" << endl;

        destroyAllWindows();
    }

    return 0;
}



// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
