#include "FeatureMatch.h"

using namespace std;
using namespace cv;


float FeatureMatch::get_coefficient(Mat matchwindow_left, Mat matchwindow_right)
{
	////Calculate correlation coefficient
	float cofficient1 = 0;
	float cofficient2 = 0;
	float cofficient3 = 0;
	for (int i = 0; i < matchwindow_left.rows; i++)
	{
		for (int j = 0; j < matchwindow_left.cols; j++)
		{
			cofficient1 += matchwindow_left.at<float>(i, j) * matchwindow_right.at<float>(i, j);
			cofficient2 += matchwindow_right.at<float>(i, j) * matchwindow_right.at<float>(i, j);
			cofficient3 += matchwindow_left.at<float>(i, j) * matchwindow_left.at<float>(i, j);
		}
	}
	float cofficient = cofficient1 / abs(sqrt(cofficient2 * cofficient3));
	return cofficient;
}


void FeatureMatch::lastview(Mat imgrgb_left, Mat imgrgb_right, vector<Point3f> feature_left, vector<Point3f> feature_right)
{
	////Combine two images
	int rows = max(imgrgb_left.rows, imgrgb_right.rows);
	Mat bothview = Mat::zeros(rows, imgrgb_left.cols + imgrgb_right.cols, imgrgb_left.type());
	for (int i = 0; i < imgrgb_left.rows; i++)
	{
		for (int j = 0; j < imgrgb_left.cols; j++)
		{
			bothview.at<Vec3b>(i, j) = imgrgb_left.at<Vec3b>(i, j);
		}
	}
	for (int i = 0; i < imgrgb_right.rows; i++)
	{
		for (int j = imgrgb_left.cols; j < imgrgb_left.cols + imgrgb_right.cols; j++)
		{
			bothview.at<Vec3b>(i, j) = imgrgb_right.at<Vec3b>(i, j - imgrgb_left.cols);
		}
	}
	////Visualize feature points and matches
	for (int i = 0; i < feature_right.size(); i++)
	{
		int a = (rand() % 200);
		int b = (rand() % 200) + 99;
		int c = (rand() % 200) - 50;
		if (a > 100 || a < 0)
		{
			a = 255;
		}
		if (b > 255 || b < 0)
		{
			b = 90;
		}
		if (c > 255 || c < 0)
		{
			c = 190;
		}
		//Draw feature points of left image
		int lx = int(feature_left.at(i).x);
		int ly = int(feature_left.at(i).y);
		circle(bothview, Point(lx, ly), 3, Scalar(0, 0, 255), 1, 4, 0);
		line(bothview, Point(lx - 3 - 2, ly), Point(lx + 3 + 2, ly), Scalar(0, 0, 255));
		line(bothview, Point(lx, ly - 3 - 2), Point(lx, ly + 3 + 2), Scalar(0, 0, 255));
		//Draw feature points of right image
		int rx = feature_right.at(i).x + imgrgb_left.cols;
		int ry = feature_right.at(i).y;
		circle(bothview, Point(rx, ry), 3, Scalar(0, 0, 255), 1, 4, 0);
		line(bothview, Point(rx - 3 - 2, ry), Point(rx + 3 + 2, ry), Scalar(0, 0, 255));
		line(bothview, Point(rx, ry - 3 - 2), Point(rx, ry + 3 + 2), Scalar(0, 0, 255));
		//Draw matches
		line(bothview, Point(lx, ly), Point(rx, ry), Scalar(0, 255, 0), 1, 8, 0);
	}
	imshow("Match Result", bothview);
	imwrite("../match.png", bothview);
	waitKey(0);
}


void FeatureMatch::correlationcoefficient_match1(string img1_pth, string img2_pth, vector<Point3f> feature_point_l, vector<Point3f> feature_point_r, int window_size, float threshold)
{
	//////Read and convert image
	Mat imgrgb_l = imread(img1_pth, IMREAD_COLOR);
	Mat imgrgb_r = imread(img2_pth, IMREAD_COLOR);
	if (imgrgb_l.empty() || imgrgb_r.empty())
	{
		cout << "Error: Invalid image path." << endl;
		return;
	}
	Mat gray_l, gray_r;
	cvtColor(imgrgb_l, gray_l, COLOR_BGR2GRAY);
	cvtColor(imgrgb_r, gray_r, COLOR_BGR2GRAY);
	//////Match with right image
	vector<Point3f> matched_point_l;
	vector<Point3f> matched_point_r;
	for (int i = 0; i < feature_point_l.size(); i++)
	{
		////Create match window of each feature point in left image
		Mat match_window_l = Mat::zeros(window_size, window_size, CV_32FC1);
		//Calculate average value of left image
		float avg_l = 0;
		if (feature_point_l.at(i).x + window_size / 2 < gray_l.cols &&
			feature_point_l.at(i).y + window_size / 2 < gray_l.rows)
		{
			for (int m = 0; m < window_size; m++)
			{
				for (int n = 0; n < window_size; n++)
				{
					avg_l += gray_l.at<uchar>(feature_point_l.at(i).y - window_size / 2 + m, feature_point_l.at(i).x - window_size / 2 + n);
					match_window_l.at<float>(m, n) = gray_l.at<uchar>(feature_point_l.at(i).y - window_size / 2 + m, feature_point_l.at(i).x - window_size / 2 + n);
				}
			}
			avg_l = avg_l / (window_size * window_size);
			//Reduce avg in left match window
			for (int m = 0; m < window_size; m++)
			{
				for (int n = 0; n < window_size; n++)
				{
					match_window_l.at<float>(m, n) -= avg_l;
				}
			}
			//////Obtain coefficient between left match window and every right window
			vector<float> match_coefficients;
			for (int j = 0; j < feature_point_r.size(); j++)
			{
				if (feature_point_r.at(j).x + window_size / 2 < gray_r.cols &&
					feature_point_r.at(j).y + window_size / 2 < gray_r.rows)
				{
					////Create match window of each feauter point in right image
					Mat match_window_r = Mat::zeros(window_size, window_size, CV_32FC1);
					//Calculate average value of left image
					float avg_r = 0;
					for (int m = 0; m < window_size; m++)
					{
						for (int n = 0; n < window_size; n++)
						{
							avg_r += gray_r.at<uchar>(feature_point_r.at(j).y - window_size / 2 + m, feature_point_r.at(j).x - window_size / 2 + n);
							match_window_r.at<float>(m, n) = gray_r.at<uchar>(feature_point_r.at(j).y - window_size / 2 + m, feature_point_r.at(j).x - window_size / 2 + n);
						}
					}
					avg_r = avg_r / (window_size * window_size);
					//Reduce avg in left match window
					for (int m = 0; m < window_size; m++)
					{
						for (int n = 0; n < window_size; n++)
						{
							match_window_r.at<float>(m, n) -= avg_l;
						}
					}
					////Obtain correlation coefficient between current right match window and left match window ,
					////according to sorted coefficients, find the best matched feature point of right image
					float coefficient = FeatureMatch::get_coefficient(match_window_l, match_window_r);
					if (coefficient > 0.7)
					{
						match_coefficients.push_back(coefficient);
					}
				}
			}
			//Find max coefficient of right windows for current left window
			float max_coefficient = 0;
			int max_index = 0;
			for (int v = 0; v < match_coefficients.size(); v++)
			{
				if (match_coefficients.at(v) > max_coefficient)
				{
					max_coefficient = match_coefficients.at(v);
					max_index = v;
				}
			}
			//Max coefficient indicate matched relationship between current left and right feature point
			//and adjust if coefficient exceed threshold
			if (max_coefficient > threshold)
			{
				matched_point_l.push_back(feature_point_l.at(i));
				matched_point_r.push_back(feature_point_r.at(max_index));
			}
		}
	}
	//////Visualize matches
	FeatureMatch::lastview(imgrgb_l, imgrgb_r, matched_point_l, matched_point_r);
}


Point3f FeatureMatch::coefficient_search_window(Mat match_window_l, Mat search_window_r)
{
	int m = match_window_l.rows;
	int n = match_window_l.cols;
	int k = search_window_r.rows;
	int l = search_window_r.cols;
	float max = 0;
	Point3f match_point;
	Mat match_window_r = Mat::zeros(m, n, search_window_r.type());
	for (int center_y = m / 2; center_y < k - m / 2; center_y++)
	{
		for (int center_x = n / 2; center_x < l - m / 2; center_x++)
		{
			float a = 0;
			float b = 0, c = 0;
			float bb = 0, cc = 0;
			float d = 0;
			float coefficient = 0;
			for (int i = -m / 2; i < m / 2; i++)
			{
				for (int j = -n / 2; j < m / 2; j++)
				{
					match_window_r.at<uchar>(i + m / 2, j + m / 2) = search_window_r.at<uchar>(center_y + i, center_x + j);
					a += match_window_l.at<uchar>(i + m / 2, j + n / 2) * match_window_r.at<uchar>(i + m / 2, j + n / 2);
					b += match_window_l.at<uchar>(i + m / 2, j + n / 2);
					bb += match_window_l.at<uchar>(i + m / 2, j + n / 2) * match_window_l.at<uchar>(i + m / 2, j + n / 2);
					c += match_window_r.at<uchar>(i + m / 2, j + n / 2);
					cc += match_window_r.at<uchar>(i + m / 2, j + n / 2) * match_window_r.at<uchar>(i + m / 2, j + n / 2);
				}
			}
			d = b * c / (m * n);
			coefficient = (a - d) / sqrt((bb - b * b / (m * n)) * (cc - c * c / (m * n)));
			if (coefficient > max)
			{
				max = coefficient;
				match_point.x = center_x;
				match_point.y = center_y;
				match_point.z = max;
			}
		}
	}
	return match_point;
}


void FeatureMatch::correlationcoefficient_match2(string img1_pth, string img2_pth, int dist_x, int dist_y, vector<Point3f> feature_point_l, int window_size, int search_size, float threshold, vector<Point3f>& matched_point_l, vector<Point3f>& matched_point_r)
{
	//////Read and convert image
	Mat imgrgb_l = imread(img1_pth, IMREAD_COLOR);
	Mat imgrgb_r = imread(img2_pth, IMREAD_COLOR);
	if (imgrgb_l.empty() || imgrgb_r.empty())
	{
		cout << "Error: Invalid image path." << endl;
		return;
	}
	Mat gray_l, gray_r;
	cvtColor(imgrgb_l, gray_l, COLOR_BGR2GRAY);
	cvtColor(imgrgb_r, gray_r, COLOR_BGR2GRAY);
	//////Match based on search area in right image
	vector<Point3f> feature_point_r;
	Point3f tmp_point_r;
	////Take estimated points(left image points + coordinates disparity) as center of search areas
	for (int i = 0; i < feature_point_l.size(); i++)
	{
		tmp_point_r.x = feature_point_l.at(i).x - dist_x;
		tmp_point_r.y = feature_point_l.at(i).y - dist_y;
		tmp_point_r.z = 0;
		//Judge if estimated points exceed the range of right image
		if (tmp_point_r.x > search_size / 2 && tmp_point_r.x < gray_r.cols - search_size / 2 &&
			tmp_point_r.y > search_size / 2 && tmp_point_r.y < gray_r.rows - search_size / 2)
		{
			feature_point_r.push_back(tmp_point_r);
		}
	}
	////Assign windows according to origin images
	//vector<Point3f> matched_point_l;
	//vector<Point3f> matched_point_r;
	Mat match_window_l = Mat::zeros(window_size, window_size, gray_l.type());
	for (int t = 0; t < feature_point_r.size(); t++)
	{
		int rx = feature_point_r.at(t).x;
		int ry = feature_point_r.at(t).y;
		int lx = feature_point_l.at(t).x;
		int ly = feature_point_l.at(t).y;

		Mat search_window_r = Mat::zeros(search_size, search_size, gray_r.type());
		//Judge if match window exceed the range of left image
		if (ly - window_size / 2 > 0 && ly + window_size / 2 < gray_l.rows && lx - window_size / 2 > 0 && lx + window_size / 2 < gray_l.cols)
		{
			for (int i = -search_size / 2; i < search_size / 2; i++)
			{
				for (int j = -search_size / 2; j < search_size / 2; j++)
				{
					//Obtain search window in right image(search_size*search_size)
					search_window_r.at<uchar>(i + search_size / 2, j + search_size / 2) = gray_r.at<uchar>(ry + i, rx + j);
				}
			}
			for (int i = -window_size / 2; i < window_size / 2; i++)
			{
				for (int j = -window_size / 2; j < window_size / 2; j++)
				{
					//Obtain match window in left image (window_size*window_size)
					match_window_l.at<uchar>(i + window_size / 2, j + window_size / 2) = gray_l.at<uchar>(ly + i, lx + j);
				}
			}
			////Find matched point in search window based on Correlation coefficient
			Point3f c = coefficient_search_window(match_window_l, search_window_r);
			//Convert to right image coordinate
			c.x = rx + search_size / 2 - c.x;
			c.y = ry + search_size / 2 - c.y;
			//Coefficient threshold restrain
			if (c.z > threshold)
			{
				matched_point_r.push_back(c);
				matched_point_l.push_back(Point3f(lx, ly, c.z));
			}
		}
	}
	//////Visualize matches
	FeatureMatch::lastview(imgrgb_l, imgrgb_r, matched_point_l, matched_point_r);
}


void FeatureMatch::leastsquare_match(string img1_pth, string img2_pth, vector<Point3f> feature_point_l, vector<Point3f> feature_point_r, int window_size)
{
	//////Read images
	Mat imgrgb_l = imread(img1_pth, IMREAD_COLOR);
	Mat imgrgb_r = imread(img2_pth, IMREAD_COLOR);
	if (imgrgb_l.empty() || imgrgb_r.empty())
	{
		cout << "Error: Invalid image path." << endl;
		return;
	}
	Mat imggray_l, imggray_r;
	cvtColor(imgrgb_l, imggray_l, COLOR_BGR2GRAY);
	cvtColor(imgrgb_r, imggray_r, COLOR_BGR2GRAY);

	vector<Point3f> matched_point_l;
	vector<Point3f> matched_point_r;
	vector<Point3f> matched_point_r_LSM;
	//////Geometric distortion initialization
	//weight matrix
	Mat formerP = Mat::eye(2 * feature_point_l.size(), 2 * feature_point_l.size(), CV_32FC1);
	//constant matrix
	Mat formerL = Mat::zeros(2 * feature_point_l.size(), 1, CV_32FC1);
	//coefficient matrix
	Mat formerA = Mat::zeros(2 * feature_point_l.size(), 6, CV_32FC1);
	for (int i = 0; i < feature_point_l.size(); i++)
	{
		float i1 = feature_point_l.at(i).y;
		float j1 = feature_point_l.at(i).x;
		float i2 = feature_point_r.at(i).y;
		float j2 = feature_point_r.at(i).x;
		float ncc = feature_point_r.at(i).z;
		formerP.at<float>(2 * i, 2 * i) = ncc;
		formerP.at<float>(2 * i + 1, 2 * i + 1) = ncc;
		formerL.at<float>(2 * i, 0) = i2;
		formerL.at<float>(2 * i + 1, 0) = j2;
		formerA.at<float>(2 * i, 0) = 1;
		formerA.at<float>(2 * i, 1) = i1;
		formerA.at<float>(2 * i, 2) = j1;
		formerA.at<float>(2 * i + 1, 3) = 1;
		formerA.at<float>(2 * i + 1, 4) = i1;
		formerA.at<float>(2 * i + 1, 5) = j1;
	}
	Mat Nbb = formerA.t() * formerP * formerA;
	Mat U = formerA.t() * formerP * formerL;
	Mat formerR = Nbb.inv() * U;
	//////Least square iteration
	for (int i = 0; i < feature_point_l.size(); i++)
	{
		////Initiate coordinate
		float i1 = feature_point_l.at(i).y;
		float j1 = feature_point_l.at(i).x;
		float i2 = feature_point_r.at(i).y;
		float j2 = feature_point_r.at(i).x;
		////Initiate Geometric distortion coefficient
		float a0 = formerR.at<float>(0, 0);
		float a1 = formerR.at<float>(1, 0);
		float a2 = formerR.at<float>(2, 0);
		float b0 = formerR.at<float>(3, 0);
		float b1 = formerR.at<float>(4, 0);
		float b2 = formerR.at<float>(5, 0);
		////Initiate Radiation distortion coefficient
		float h0 = 0;
		float h1 = 1;
		////Iteration
		////(Iteration exit conditions): the latter correlation coefficient is smaller than the previous one
		float beforencc = feature_point_l.at(i).z, ncc = feature_point_l.at(i).z;
		float is = 0, js = 0;
		while (beforencc <= ncc)
		{
			beforencc = ncc;
			//coefficient matrix (V = CX-L)
			Mat C = Mat::zeros(window_size * window_size, 8, CV_32FC1);
			//constant matrix
			Mat L = Mat::zeros(window_size * window_size, 1, CV_32FC1);
			//weight matrix
			Mat P = Mat::eye(window_size * window_size, window_size * window_size, CV_32FC1);
			float sum_gi_sq = 0, sum_gj_sq = 0, sum_igi_sq = 0, sum_jgj_sq = 0;
			int dimension = 0;
			float sum_win_l = 0, sum_win_l_sq = 0, sum_win_r = 0, sum_win_r_sq = 0, sum_lr = 0;
			for (int m = i1 - window_size / 2; m <= i1 + window_size / 2; m++)
			{
				for(int n = j1 - window_size / 2; n <= j1 + window_size / 2; n++)
				{
					if (m > 0 && m < imggray_r.rows && n> 0 && n < imggray_r.cols)
					{
						float i2 = a0 + a1 * m + a2 * n;
						float j2 = b0 + b1 * m + b2 * n;
						int I = floor(i2);
						int J = floor(j2);
						if (I > 1 && I < imggray_r.rows - 1 && J> 1 && J < imggray_r.cols - 1)
						{
							//Bilinear interpolation resampling
							float lineGray = (J + 1 - j2) * ((I + 1 - i2) * imggray_r.at<uchar>(I, J) + (i2 - I) * imggray_r.at<uchar>(I + 1, J)) + (j2 - J) * ((I + 1 - i2) * imggray_r.at<uchar>(I, J + 1) + (i2 - I) * imggray_r.at<uchar>(I + 1, J + 1));
							//Radiation calibration
							float radioGray = h0 + h1 * lineGray;
							sum_win_r += radioGray;
							sum_win_r_sq += radioGray * radioGray;
							//Calculate coefficient matrix C (V = CX -L)
							float gj = 0.5 * (imggray_r.at<uchar>(I, J + 1) - imggray_r.at<uchar>(I, J - 1));
							float gi = 0.5 * (imggray_r.at<uchar>(I + 1, J) - imggray_r.at<uchar>(I - 1, J));
							C.at<float>(dimension, 0) = 1;
							C.at<float>(dimension, 1) = lineGray;
							C.at<float>(dimension, 2) = gi;
							C.at<float>(dimension, 3) = i2 * gi;
							C.at<float>(dimension, 4) = j2 * gi;
							C.at<float>(dimension, 5) = gj;
							C.at<float>(dimension, 6) = i2 * gj;
							C.at<float>(dimension, 7) = j2 * gj;
							//Calculate constant matrix L
							L.at<float>(dimension, 0) = imggray_l.at<uchar>(m, n) - radioGray;
							dimension += 1;
							//Obtain sum for calculating best match position
							float gj_l = 0.5 * (imggray_l.at<uchar>(m, n + 1) - imggray_l.at<uchar>(m, n - 1));
							float gi_l = 0.5 * (imggray_r.at<uchar>(m + 1, n) - imggray_r.at<uchar>(m - 1, n));
							sum_gi_sq += gi_l * gi_l;
							sum_gj_sq += gj_l * gj_l;
							sum_igi_sq += m * gi_l * gi_l;
							sum_jgj_sq += n * gj_l * gj_l;
							//Obtain sum for calculating NCC
							sum_win_l += imggray_l.at<uchar>(m, n);
							sum_win_l_sq += imggray_l.at<uchar>(m, n) * imggray_l.at<uchar>(m, n);
							sum_lr += radioGray * imggray_l.at<uchar>(m, n);
						}
					}
				}
			}
			//Calculate NCC
			float coefficient1 = sum_lr - sum_win_l * sum_win_r / (window_size * window_size);
			float coefficient2 = sum_win_l_sq - sum_win_l * sum_win_l / (window_size * window_size);
			float coefficient3 = sum_win_r_sq - sum_win_r * sum_win_r / (window_size * window_size);
			ncc = coefficient1 / sqrt(coefficient2 * coefficient3);
			//Calculate Geometric and Radiation coefficient
			Mat Nb = C.t() * P * C, Ub = C.t() * P * L;
			Mat para = Nb.inv() * Ub;
			float dh0 = para.at<float>(0, 0);
			float dh1 = para.at<float>(1, 0);
			float da0 = para.at<float>(2, 0);
			float da1 = para.at<float>(3, 0);
			float da2 = para.at<float>(4, 0);
			float db0 = para.at<float>(5, 0);
			float db1 = para.at<float>(6, 0);
			float db2 = para.at<float>(7, 0);
			a0 = a0 + da0 + a0 * da1 + b0 * da2;
			a1 = a1 + a1 * da1 + b1 * da2;
			a2 = a2 + a2 * da1 + b2 * da2;
			b0 = b0 + db0 + a0 * db1 + b0 * db2;
			b1 = b1 + a1 * db1 + b1 * db2;
			b2 = b2 + a2 * db1 + b2 * db2;
			h0 = h0 + dh0 + h0 * dh1;
			h1 = h1 + h1 * dh1;
			//Calculate best match position
			float it = sum_igi_sq / sum_gi_sq;
			float jt = sum_jgj_sq / sum_gj_sq;
			is = a0 + a1 * it + a2 * jt;
			js = b0 + b1 * it + b2 * jt;
		}
		Point3f tmp;
		tmp.x = js;
		tmp.y = is;
		tmp.z = ncc;
		if (tmp.y > 0 && tmp.y < imggray_r.rows && tmp.x> 0 && tmp.x < imggray_r.cols && abs(tmp.y - feature_point_r.at(i).y) < 5 && abs(tmp.x - feature_point_r.at(i).x) < 5)
		{
			matched_point_l.push_back(feature_point_l.at(i));
			matched_point_r.push_back(feature_point_r.at(i));
			matched_point_r_LSM.push_back(tmp);
		}
	}
	//////Visualize
	lastview(imgrgb_l, imgrgb_r, matched_point_l, matched_point_r_LSM);
	ofstream outputfile;
	outputfile.open("../../matchedright.txt");
	if (outputfile.is_open())
	{
		outputfile << "Origin points " << feature_point_l.size() << "->Adjusted points " << matched_point_r_LSM.size() << endl;
		for (int i = 0; i < matched_point_r_LSM.size(); i++)
		{
			outputfile << "(" << matched_point_r.at(i).x << ", " << matched_point_r.at(i).y << ")" << "(" << matched_point_r_LSM.at(i).x << ", " << matched_point_r_LSM.at(i).y << ")" <<endl;
		}
	}
	outputfile.close();
}	
