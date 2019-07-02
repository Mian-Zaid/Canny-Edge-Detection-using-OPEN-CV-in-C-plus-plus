

#include<iostream>
#include<cmath>
#include <stdlib.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

// Global Variables...


Mat gray, rgb, dst, dst_2, detected_edges, binary, dilated, grayWithNoise,contours;

int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;

int max_thresh = 255;

Mat  x2y2, xy, mtrace, x_derivative, y_derivative, x2_derivative, y2_derivative,
xy_derivative, x2g_derivative, y2g_derivative, xyg_derivative, Dst, Dst_norm, Dst_norm_scaled;
int thresh = 128;

///////////////////////////////SOBEL edge detection/////////////////////////////////

// Computes the x component of the gradient vector at a given point in a image.
//returns gradient in the x direction
int xGradient(Mat image, int x, int y)
{
	return image.at<uchar>(y - 1, x - 1) +
		2 * image.at<uchar>(y, x - 1) +
		image.at<uchar>(y + 1, x - 1) -
		image.at<uchar>(y - 1, x + 1) -
		2 * image.at<uchar>(y, x + 1) -
		image.at<uchar>(y + 1, x + 1);
}

// Computes the y component of the gradient vector at a given point in a image
// returns gradient in the y direction

int yGradient(Mat image, int x, int y)
{
	return image.at<uchar>(y - 1, x - 1) +
		2 * image.at<uchar>(y - 1, x) +
		image.at<uchar>(y - 1, x + 1) -
		image.at<uchar>(y + 1, x - 1) -
		2 * image.at<uchar>(y + 1, x) -
		image.at<uchar>(y + 1, x + 1);
}


void onTrackbar(int, void*) {
	//cvtColor(src, src_gray, CV_BGR2GRAY);
	Mat harrisOut = contours.clone();
	//Step one
	//to calculate x and y derivative of image we use Sobel function
	//Sobel( srcimage, dstimage, depthofimage -1 means same as input, xorder 1,yorder 0,kernelsize 3, BORDER_DEFAULT);
	Sobel(harrisOut, x_derivative, CV_32FC1, 1, 0, 3, BORDER_DEFAULT);
	Sobel(harrisOut, y_derivative, CV_32FC1, 0, 1, 3, BORDER_DEFAULT);
	//Step Two calculate other three images in M
	pow(x_derivative, 2.0, x2_derivative);
	pow(y_derivative, 2.0, y2_derivative);
	multiply(x_derivative, y_derivative, xy_derivative);
	//step three apply gaussain
	GaussianBlur(x2_derivative, x2g_derivative, Size(7, 7), 2.0, 0.0, BORDER_DEFAULT);
	GaussianBlur(y2_derivative, y2g_derivative, Size(7, 7), 0.0, 2.0, BORDER_DEFAULT);
	GaussianBlur(xy_derivative, xyg_derivative, Size(7, 7), 2.0, 2.0, BORDER_DEFAULT);
	//forth step calculating R with k=0.04
	multiply(x2g_derivative, y2g_derivative, x2y2);
	multiply(xyg_derivative, xyg_derivative, xy);
	pow((x2g_derivative + y2g_derivative), 2.0, mtrace);
	Dst = (x2y2 - xy) - 0.04 * mtrace;
	//normalizing result from 0 to 255
	normalize(Dst, Dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(Dst_norm, Dst_norm_scaled);
	// Drawing a circle around corners
	for (int j = 0; j < harrisOut.rows; j++)
	{
		for (int i = 0; i < harrisOut.cols; i++)
		{
			if ((int)Dst_norm.at<float>(j, i) > thresh)
			{
				circle(harrisOut, Point(i, j), 5, Scalar(255), 2, 8, 0);
			}
		}
	}
	// Showing the result
	//namedWindow(corners_window, CV_WINDOW_AUTOSIZE);
	imshow("Harris Cornners", harrisOut);
}

//sort the window using insertion sort
//insertion sort is best for this sorting
void insertionSort(int window[])
{
	int temp, i, j;
	for (i = 0; i < 9; i++) {
		temp = window[i];
		for (j = i - 1; j >= 0 && temp < window[j]; j--) {
			window[j + 1] = window[j];
		}
		window[j + 1] = temp;
	}
}

void NoiseRemovel(Mat grayWithNoise,Mat &output) {
	//create a sliding window of size 9
	int window[9];

	output = grayWithNoise.clone();
	for (int y = 0; y < grayWithNoise.rows; y++)
		for (int x = 0; x < grayWithNoise.cols; x++)
			output.at<uchar>(y, x) = 0.0;

	for (int y = 1; y < grayWithNoise.rows - 1; y++) {
		for (int x = 1; x < grayWithNoise.cols - 1; x++) {

			// Pick up window element

			window[0] = grayWithNoise.at<uchar>(y - 1, x - 1);
			window[1] = grayWithNoise.at<uchar>(y, x - 1);
			window[2] = grayWithNoise.at<uchar>(y + 1, x - 1);
			window[3] = grayWithNoise.at<uchar>(y - 1, x);
			window[4] = grayWithNoise.at<uchar>(y, x);
			window[5] = grayWithNoise.at<uchar>(y + 1, x);
			window[6] = grayWithNoise.at<uchar>(y - 1, x + 1);
			window[7] = grayWithNoise.at<uchar>(y, x + 1);
			window[8] = grayWithNoise.at<uchar>(y + 1, x + 1);

			// sort the window to find median
			insertionSort(window);

			// assign the median to centered element of the matrix
			output.at<uchar>(y, x) = window[4];
		}
	}


}


int main()
{


	int gx, gy, sum;


	// Load an image
	rgb = imread("bottle_cap_4.jpg");
	if (!rgb.data)
	{
		cout << "Error while loading an image \n";
		return -1;
	}
	cvtColor(rgb, grayWithNoise, COLOR_RGB2GRAY);


	//////////////////Gray Noise Removal/////////////

	NoiseRemovel(grayWithNoise, gray);

	///////////////////////binary conversion//////////////////////////

	adaptiveThreshold(gray, binary, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 105, 1);


	dst = gray.clone();
	if (!gray.data)
	{
		return -1;
	}


	///////////////////////////////Sobel Filter///////////////////////////

	for (int y = 0; y < gray.rows; y++)
		for (int x = 0; x < gray.cols; x++)
			dst.at<uchar>(y, x) = 0.0;

	for (int y = 1; y < gray.rows - 1; y++) {
		for (int x = 1; x < gray.cols - 1; x++) {
			gx = xGradient(gray, x, y);
			gy = yGradient(gray, x, y);
			sum = abs(gx) + abs(gy);
			sum = sum > 255 ? 255 : sum;
			sum = sum < 0 ? 0 : sum;
			dst.at<uchar>(y, x) = sum;
		}
	}





	///////////////////////////////////// CANNY  edge detection////////////////////////////////

	Canny(gray, contours, 50,50);
	
	Mat cannyAfterNoise;
	NoiseRemovel(contours, cannyAfterNoise);
	
	
	////////////////////////////////// DIlation /////////////////////////

	dilate(contours, dilated, Mat(), Point(-1, -1), 2, 1, 1);

	///////////////////////////// fill holes /////////////////////////

	// Floodfill from point (0, 0)
	Mat im_floodfill = dilated.clone();
	floodFill(im_floodfill, cv::Point(0, 0), Scalar(255));
	//imshow("floodfill", im_floodfill);

	// Invert floodfilled image
	Mat im_floodfill_inv;
	bitwise_not(im_floodfill, im_floodfill_inv);
	//imshow("floodfill invers", im_floodfill_inv);

	// Combine the two images to get the foreground.
	Mat im_out = (dilated | im_floodfill_inv);
	//imshow("Foreground", im_out);


	 //////////////////////////////// HARIS ////////////////////////////////

	createTrackbar("Threshold", "Harris Cornners", &thresh, 255, onTrackbar);

	onTrackbar(thresh, 0);
	

	/////////////////////////////////// show all images //////////////////


	imshow("Original Image", rgb);
	imshow("Binary image", binary);
	imshow("Gray with noise", gray);
	imshow("Gray After Noise Removal", gray);
	
	imshow("Sobel", dst);
	imshow("Canny", contours);
	imshow("Canny after Dilated", dilated);
	//imshow("canny after noise removal", cannyAfterNoise);
	
	


	waitKey(0);
	return 0;
}