#pragma once
#pragma once
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <cmath>

using namespace std;
using namespace cv;


class FeatureMatch
{
public:
	static float get_coefficient(Mat matchwindow_left, Mat matchwindow_right);
	static void lastview(Mat imgrgb_left, Mat imgrgb_right, vector<Point3f> feature_left, vector<Point3f> feature_right);
	static void correlationcoefficient_match1(string img1_pth, string img2_pth, vector<Point3f> feature_point_left, vector<Point3f> feature_point_right, int window_size, float threshold);
	static Point3f coefficient_search_window(Mat match_window_left, Mat search_window_right);
	static void correlationcoefficient_match2(string img1_pth, string img2_pth, int dist_x, int dist_y, vector<Point3f> feature_point_left, int window_size, int search_size, float threshold, vector<Point3f>& matched_point_left, vector<Point3f>& matched_point_right);
	static void leastsquare_match(string img1_pth, string img2_pth, vector<Point3f> feature_point_left, vector<Point3f> feature_point_right, int window_size);
};


