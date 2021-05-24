#include <vector>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include "DOMAIN_TRANSFORM.h"
#include "TIME_UTILITIES.h"
#include "SETTINGS.h"

using namespace cv;
using namespace std;
using namespace CFL2;

int sigma_D, sigma_S, sigma_R;
SETTINGS settings;
IplImage *src, *depth;
DOMAIN_TRNASFORM dt;

void trackbar_sigma_d(int, void*)
{	
	settings.sigma_d = ((double)sigma_D / 400);
	cout << "PROCESSING " << endl;
	IplImage *roll = dt.Rolling_Guidance(&settings, src, depth);

	imshow("Rolling", Mat(roll));	
}

void trackbar_sigma_s(int, void*)
{
	settings.sigma_s = ((double)sigma_S / 10);
	cout << "PROCESSING " << endl;
	IplImage *roll = dt.Rolling_Guidance(&settings, src, depth);

	imshow("Rolling", Mat(roll));
}

void trackbar_sigma_r(int, void*)
{
	settings.sigma_r = ((double)sigma_R / 400);
	cout << "PROCESSING " << endl;
	IplImage *roll = dt.Rolling_Guidance(&settings, src, depth);

	imshow("Rolling", Mat(roll));
}

int main(int argc,char** argv)
{	
	settings.sigma_s = 40;
	settings.sigma_r = 0.5;
	settings.sigma_d = 0.3;
	settings.sigma_h = settings.sigma_s;
	settings.num_iterDT = 5;
	settings.num_iterRoll = 5;
	settings.focalLength = 35;
	
	int number_of_mp=4;
	omp_set_num_threads(number_of_mp);
	src = cvLoadImage("color90.png", 1);
	depth = cvLoadImage("depth90.png", 0);
		
	std::cout<<"Number of omp := "<<omp_get_max_threads()<<std::endl;
	
	IplImage *res = dt.Normalized_Convolution(src, 40, 0.5, 40, 5);
	
	//IplImage *roll = dt.Rolling_Guidance(&settings);
	
	cvNamedWindow("Result", CV_WINDOW_AUTOSIZE);
	namedWindow("Rolling", 1);
	cvNamedWindow("Original", CV_WINDOW_AUTOSIZE);
	cvShowImage("Result", res);
	cvShowImage("Original", src);
	//cvShowImage("Rolling", roll);
	

	createTrackbar("Sigma_d", "Rolling", &sigma_D, 1000, trackbar_sigma_d);
	createTrackbar("Sigma_r", "Rolling", &sigma_R, 1000, trackbar_sigma_r);
	createTrackbar("Sigma_s", "Rolling", &sigma_S, 1000, trackbar_sigma_s);
	cvWaitKey(0);



	cvReleaseImage(&src);
	cvReleaseImage(&res);
	//cvReleaseImage(&roll);

	return 0;
}
