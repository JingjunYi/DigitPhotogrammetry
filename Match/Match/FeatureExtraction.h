#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <cmath>

using namespace std;
using namespace cv;


class FeatureExtraction
{
public:
	static void siftfeature_extraction(string img_pth, int point_num, vector<Point3f>& feature);

	static void find_minmax(float a[], int m, float& max, float& min);
	static void moravecfeature_extraction(string img_pth, int window_size, int nms_size, float threshold, vector<Point3f>& feature);
};
