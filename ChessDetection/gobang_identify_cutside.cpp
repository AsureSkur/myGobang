#include <iostream>
#include <cstdio>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2\imgproc\types_c.h>
using namespace cv;
#define TESTNUM 10

struct HoughVal
{
	double step;
	double rest;
	int param1;
	int param2;
};

void HoughCheckChess(const char* imgname, std::vector<cv::Vec3f>& circles, struct HoughVal value)
{
	Mat srcImage = imread(imgname);
	Mat midImage;
	Mat dstImage;
	int minDst = (srcImage.rows + srcImage.cols) / (value.step + 14) / 4 - 10;
	int minRadius = minDst;
	if (minRadius < 0) minRadius = 0;
	int maxRadius = (srcImage.rows + srcImage.cols) / (value.step + 14) / 2 + 10;

	GaussianBlur(srcImage, srcImage, Size(9, 9), 1, 1);
	Laplacian(srcImage, midImage, srcImage.depth());
	add(srcImage, midImage, midImage);
	cvtColor(midImage, dstImage, COLOR_BGR2GRAY);
	//Laplacian(midImage, midImage, midImage.depth());
	//GaussianBlur(midImage, midImage, Size(9, 9), 2, 2);
	//   medianBlur(cimg, cimg, 5);
	//Canny(midImage, midImage, 10, 250, 5);

	HoughCircles(dstImage, circles, CV_HOUGH_GRADIENT, 1.0, minDst, value.param1, value.param2, minRadius, maxRadius);

	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		circle(srcImage, center, 3, Scalar(0, 255, 0));
		circle(srcImage, center, radius, Scalar(155, 50, 255));
	}
	//imshow(imgname, srcImage);
	//imshow(imgname, dstImage);
	//waitKey();
	//cv::destroyWindow(imgname);
	char filepath[260]="result/";
	strcat(filepath, imgname);
	imwrite(filepath, srcImage);
	return;
}
void posCheckChess(cv::InputArray src, std::vector<cv::Vec3f>& circles, int(*chessboard)[15])
{
	int rows = 15;
	int x, y;
	double x0, y0;
	for (int i = 0; i < circles.size(); i++)
	{
		x0 = circles[i][0] * 14 / src.rows();
		y0 = circles[i][1] * 14 / src.cols();
		x = x0 / 1;
		y = y0 / 1;
		double cmp[4];
		cmp[0] = std::sqrt((x0 - x) * (x0 - x) + (y0 - y) * (y0 - y));
		cmp[1] = std::sqrt((x0 - x - 1) * (x0 - x - 1) + (y0 - y) * (y0 - y));
		cmp[2] = std::sqrt((x0 - x - 1) * (x0 - x - 1) + (y0 - y - 1) * (y0 - y - 1));
		cmp[3] = std::sqrt((x0 - x) * (x0 - x) + (y0 - y - 1) * (y0 - y - 1));
		int min = 0;
		for (int i = 1; i < 4; i++)
		{
			if (cmp[i] < cmp[min])
			{
				min = i;
			}
		}
		if (cmp[min] < 0.5)
		{
			switch (min)
			{
			case 0: chessboard[x][y] = 1; break;
			case 1: chessboard[x + 1][y] = 1; break;
			case 2: chessboard[x + 1][y + 1] = 1; break;
			case 3: chessboard[x][y + 1] = 1; break;
			}
		}

	}
	return;
}
int evaluateFunc(int(*chessboard)[15], int serial, int prednum)
{
	int loss;
	int miss = 0;
	int labelnum = 0;
	//if miss,loss++;if wrong, loss++;else go ahead
	char filepath[260] = "data/label";
	char tmp[20];
	itoa(serial, tmp, 10);
	strcat(filepath, tmp);
	strcat(filepath, ".txt");
	FILE* fp = freopen(filepath, "r", stdin);
	int x, y, type;
	//compare label and predict
	while (scanf("%d %d %d ", &x, &y, &type) != EOF)
	{
		labelnum++;
		if (!chessboard[x][y])
		{
			miss++;
		}
	}

	loss = prednum - labelnum + miss * 2;
	fclose(stdin);
	return loss;
}
void gradientDescent(int loss, int lastloss, struct HoughVal& val, struct HoughVal& lastval)
{
	double rate = 0.1;
	if (loss == 0)
	{
		return;
	}
	if (loss == lastloss)
	{
		val.step += (rand() % 3 - (double)1) * rate;
		val.rest += (rand() % 3 - (double)1) * rate;
		val.param1 += rand() % 3 - 1;
		val.param2 += rand() % 3 - 1;
		return;
	}
	int deltalost = loss - lastloss;
	double delta[4];
	double tmp = 0.0;
	delta[0] = val.step - lastval.step;
	delta[1] = val.rest - lastval.rest;
	delta[2] = (double)val.param1 - lastval.param1;
	delta[3] = (double)val.param2 - lastval.param2;
	lastval.step = val.step;
	lastval.rest = val.rest;
	lastval.param1 = val.param1;
	lastval.param2 = val.param2;
	if (delta[0])
	{
		tmp = rate * deltalost / delta[0];
		while (fabs(tmp) > 1)
		{
			tmp /= 10;
		}
		//val.step -= (floor(rate * deltalost / delta[0]) > 1) ? floor(rate * deltalost) / delta[0] : 1;
		val.step -= tmp;
	}
	else val.step += (rand() % 3 - (double)1) * rate;
	if (delta[1])
	{
		tmp = rate * deltalost / delta[1];
		while (fabs(tmp) > 1)
		{
			tmp /= 10;
		}
		//val.rest -= (floor(rate * deltalost) / delta[1] > 1) ? floor(rate * deltalost) / delta[1] : 1;
		val.rest -= tmp;
	}
	else val.rest += (rand() % 3 - (double)1) * rate;
	if (delta[2])
	{
		tmp = rate * deltalost / delta[2];
		while (fabs(tmp) > 10)
		{
			tmp /= 10;
		}
		val.param1 -= tmp;
		//val.param1 -= (floor(rate * deltalost) / delta[2] > 1) ? floor(rate * deltalost) / delta[2] : 1;
	}
	else val.param1 += rand() % 3 - 1;
	if (delta[3])
	{
		tmp = rate * deltalost / delta[3];
		while (fabs(tmp) > 10)
		{
			tmp /= 10;
		}
		val.param2 -= tmp;
		//val.param2 -= (floor(rate * deltalost) / delta[3] > 1) ? floor(rate * deltalost) / delta[3] : 1;
	}
	else val.param2 += rand() % 3 - 1;

	//avoid becoming zero
	if (val.step <= 0)val.step = 1;
	if (val.rest <= 0)val.rest = 1;
	if (val.param1 <= 0)val.param1 = 1;
	if (val.param2 <= 0)val.param2 = 1;
	return;
}

int main(int argc, const char* argv[])
{
	struct HoughVal val, lastval;
	std::vector<cv::Vec3f>circles[TESTNUM];
	char head[260] = "data/";
	char filepath[260] = "";
	//initial the value for houghcircle calculator
	val.step = 1;
	val.rest = 0;
	val.param1 = 100;
	val.param2 = 20;
	lastval.step = val.step;
	lastval.rest = val.rest;
	lastval.param1 = val.param1;
	lastval.param2 = val.param2;
	int loss = 0;
	int lastloss = 0;
	int clk = 0;
	int times = 0;
	while (1)
	{
		printf("Times:%d\nstep:%lf\nrest:%lf\npara1:%d\npara2:%d\n", times, val.step, val.rest, val.param1, val.param2);
		times++;
		if (clk > 500)
		{
			break;
		}
		loss = 0;
		for (int i = 1; i <= TESTNUM; i++)
		{
			char tmp[20];
			memset(filepath, 0, sizeof(filepath));
			itoa(i, tmp, 10);
			strcpy(filepath, head);
			strcat(filepath, tmp);
			strcat(filepath, ".png");
			HoughCheckChess(filepath, circles[i - 1], val);
			//cv::Mat chessboard = cv::Mat::zeros(15, 15, CV_8U);
			int chessboard[15][15] = { 0 };
			posCheckChess(imread(filepath), circles[i - 1], chessboard);
			loss += evaluateFunc(chessboard, i, circles[i - 1].size());
		}
		gradientDescent(loss, lastloss, val, lastval);
		lastloss = loss;
		printf("loss:%d\n\n", loss);
		if (loss <= 1)
		{
			break;
		}
		clk++;
	}
	printf("final:\nstep:%lf\nrest:%lf\npara1:%d\npara2:%d\n", val.step, val.rest, val.param1, val.param2);

	return 0;
}