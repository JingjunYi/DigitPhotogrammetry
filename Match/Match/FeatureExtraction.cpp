#include "FeatureExtraction.h"

using namespace std;
using namespace cv;


void FeatureExtraction::siftfeature_extraction(string img1_pth, int point_num, vector<Point3f>& feature)
{
	////Read images
	Mat img1 = imread(img1_pth);
	if (img1.empty())
	{
		cout << "Error: Invalid image path." << endl;
		return;
	}
	////Sift feature detection
	vector<KeyPoint> keypoint1;
	Mat descriptor1;
	Ptr<Feature2D> siftfeature = SIFT::create(point_num);
	siftfeature->detect(img1, keypoint1);
	//cout << "Feature detected." << endl;
	siftfeature->compute(img1, keypoint1, descriptor1);
	//cout << "Descriptor computed." << endl;
	////Paint feature points
	//Mat feature_img1;
	//drawKeypoints(img1, keypoint1, feature_img1);
	//imshow("feature1", feature_img1);
	//waitKey(0);
	Point3f feature_point;
	for (int i = 0; i < keypoint1.size(); i++)
	{
		feature_point.x = keypoint1[i].pt.x;
		feature_point.y = keypoint1[i].pt.y;
		feature_point.z = keypoint1[i].response;
		feature.push_back(feature_point);
		//Visualize
		circle(img1, Point(int(feature_point.x), int(feature_point.y)), 3, Scalar(0, 0, 255), 1, 4, 0);
		line(img1, Point(int(feature_point.x) - 3 - 2, int(feature_point.y)), Point(int(feature_point.x) + 3 + 2, int(feature_point.y)), Scalar(0, 0, 255), 1, 8, 0);
		line(img1, Point(int(feature_point.x), int(feature_point.y) - 3 - 2), Point(int(feature_point.x), int(feature_point.y) + 3 + 2), Scalar(0, 0, 255), 1, 8, 0);
	}
	imshow("Sift Feature", img1);
	waitKey(0);
	imwrite("../siftfeature.png", img1);
}


void FeatureExtraction::find_minmax(float a[], int m, float& max, float& min)
{
	min = a[0];
	max = a[0];
	for (int i = 0; i < m; i++)
	{
		if (a[i] > max)
		{
			max = a[i];
			continue;
		}
		else if (a[i] < min)
		{
			min = a[i];
			continue;
		}
	}
}


void FeatureExtraction::moravecfeature_extraction(string img_pth, int window_size, int nms_size, float threshold, vector<Point3f>& feature)
{
	////Read images
	Mat imgrgb = imread(img_pth);
	if (imgrgb.empty())
	{
		cout << "Error: Invalid image path." << endl;
		return;
	}
	////Gray and Blur
	Mat gray;
	cvtColor(imgrgb, gray, COLOR_RGB2GRAY);
	Mat img;
	GaussianBlur(gray, img, Size(5, 5), 0, 0);
	////Obtain interest value	on four directions
	Mat interest = Mat::zeros(img.rows, img.cols, CV_32FC1);
	Mat candidate(img.rows, img.cols, img.type());
	for (int i = window_size / 2; i < img.rows - window_size / 2; i++)
	{
		for (int j = window_size / 2; j < img.cols - window_size / 2; j++)
		{
			float d[4];
			d[0] = d[1] = d[2] = d[3] = 0;
			for (int k = -1 * window_size / 2; k < window_size / 2 - 1; k++)
			{
				d[0] += pow(float(img.at<uchar>(i + k, j)) - float(img.at<uchar>(i + k + 1, j)), 2);
				d[1] += pow(float(img.at<uchar>(i + k, j + k)) - float(img.at<uchar>(i + k + 1, j + k + 1)), 2);
				d[2] += pow(float(img.at<uchar>(i, j + k)) - float(img.at<uchar>(i, j + k + 1)), 2);
				d[3] += pow(float(img.at<uchar>(i + k, j - k)) - float(img.at<uchar>(i + k + 1, j - k - 1)), 2);
			}
			float min = 0;
			float max = 0;
			FeatureExtraction::find_minmax(d, 4, max, min);
			interest.at<float>(i, j) = min;
			////Get candidate points according to threshold
			if (interest.at<float>(i, j) >= threshold)
			{
				//Only take points with interest value over threshold (0-balck)
				candidate.at<uchar>(i, j) = 0; 
			}
			else
			{
				candidate.at<uchar>(i, j) = 255;
			}
		}
	}
	////Non-Maximum Suppression
	Point3f feature_point;
	for (int i = nms_size / 2; i < img.rows - nms_size / 2; i++)
	{
		for (int j = nms_size / 2; j < img.cols - nms_size / 2; j++)
		{
			//Only take valid candidate points
			if (candidate.at<uchar>(i, j) == 0)
			{
				float max = 0;
				//Find point with the largest interest in nms window
				for (int m = -nms_size / 2; m <= nms_size / 2; m++)
				{
					for (int n = -nms_size / 2; n <= nms_size / 2; n++)
					{
						if (interest.at<float>(i + m, j + n) > max)
						{
							max = interest.at<float>(i + m, j + n);
						}
					}
				}
				//If center of window is max consider it as feature point, otherwise ignore
				if (interest.at<float>(i, j) == max)
				{
					feature_point.x = j;
					feature_point.y = i;
					feature_point.z = interest.at<float>(i, j);
					feature.push_back(feature_point);
					//visualize
					circle(imgrgb, Point(j, i), 3, Scalar(0, 0, 255), 1, 4, 0);
					line(imgrgb, Point(j - 3 - 2,i), Point(j + 3 + 2, i), Scalar(0, 0, 255), 1, 8, 0);
					line(imgrgb, Point(j, i - 3 - 2), Point(j, i + 3 + 2), Scalar(0, 0, 255), 1, 8, 0);
				}
			}
		}
	}
	//imshow("Candidate Points", candidate);
	//waitKey(0);
	imshow("Moravec Feature", imgrgb);
	waitKey(0);
	imwrite("../moravecfeature.png", imgrgb);
}