// TargetDetectionAndTracking.cpp : 定义控制台应用程序的入口点。

#include <stdio.h>
#include <iostream>
#include <stdint.h>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <highgui/highgui.hpp>
#include <features2d/features2d.hpp>
#include <nonfree/nonfree.hpp>
#include <vector>
#include <time.h> 

#include "camshiftdemo.h"

using namespace std;
using namespace cv;

typedef std::vector<cv::KeyPoint> KeyPointVector;
typedef std::vector<cv::Point2f> Point2fVector;
typedef std::vector<cv::DMatch> DMatchVector;
typedef std::vector<cv::Mat> MatVector;
typedef std::vector<cv::Rect> RectVector;
//#define MAX(a,b) ((a)>(b)?(a):(b))
//#define MIN(a,b) ((a)<(b)?(a):(b))

#define PAUSE system("pause")

#define USE_SIFT 1

#define USET_SAUSAC (0&&USE_SIFT)

#define PYR_DOWN_UP 0

#define USE_TRACERING 1

#define VIDEO_FRAME_INTERVAL_DETECTIVE 4
#define VIDEO_FRAME_INTERVAL_TRACE 1
#define MAX_FRAME 3

#define CONTOUR_MAX_AERA 8000
#define CONTOUR_MIN_AERA 250

int iDelay = 10;
const double MHI_DURATION = 1;


void equalizeHistColorImage(cv::Mat src, cv::Mat &dst);
void printImage(cv::Mat image, char *file);
void threshold(const cv::Mat &src, cv::Mat &dst, double thresh, double maxVal, int thresholdType);
void absdiff(const cv::Mat &src1, const cv::Mat &src2, cv::Mat &dst);

char g_ImgName[100] = "C:\\Users\\liupengfei\\Desktop\\data(4.23)\\data(4.23)\\data2\\imgs\\";
//char g_ImgName[100] = "C:\\Users\\liupengfei\\Desktop\\data(4.23)\\data(4.23)\\egtest02\\imgs\\";
int readImage(int iImageIndex, cv::Mat &src_frame, int flags = cv::IMREAD_GRAYSCALE)
{
	char imgName[100] = {0};
	sprintf(imgName,"%simg%05d.jpg",g_ImgName,iImageIndex);
	//sprintf(imgName,"%sframe%05d.jpg",g_ImgName,iImageIndex*VIDEO_FRAME_INTERVAL + 1);
	src_frame = cv::imread(imgName, flags);
	if( !src_frame.data )
		return -1;
	return 0;
}

// 彩色图像二值化
void threshold(const cv::Mat &src, cv::Mat &dst, double thresh, double maxVal, int thresholdType)
{
	assert(thresholdType == cv::THRESH_BINARY || thresholdType == cv::THRESH_BINARY_INV || thresholdType == cv::THRESH_TRUNC \
		|| thresholdType == cv::THRESH_TOZERO || thresholdType == cv::THRESH_TOZERO_INV);

	assert(src.depth() == CV_8U && maxVal <= 255 && maxVal >= 0);

	dst.create(src.size(),src.type());
	switch (thresholdType) {  
	case cv::THRESH_BINARY:  
		for (int row = 0; row < src.rows;row++) {  
			for (int col = 0; col < src.cols; col++)  
			{
				int B = (*(src.data + src.step[0] * row + src.step[1] * col + src.elemSize1() * 0));
				int R = (*(src.data + src.step[0] * row + src.step[1] * col + src.elemSize1() * 1));
				int G = (*(src.data + src.step[0] * row + src.step[1] * col + src.elemSize1() * 2));
				if(pow(B,2.0)+pow(R,2.0)+pow(G,2.0) > pow(thresh,2.0))
					memset(dst.data + dst.step[0] * row + dst.step[1] * col,maxVal,sizeof(unsigned char)*dst.channels());
				else 
					memset(dst.data + dst.step[0] * row + dst.step[1] * col,0,sizeof(unsigned char)*dst.channels());
			}
		}  
		break;  

	case cv::THRESH_BINARY_INV:  
		for (int row = 0; row < src.rows;row++) {  
			for (int j = 0; j < src.cols; j++)  
			{
				int B = (*(src.data + src.step[0] * row + src.step[1] * j + src.elemSize1() * 0));
				int R = (*(src.data + src.step[0] * row + src.step[1] * j + src.elemSize1() * 1));
				int G = (*(src.data + src.step[0] * row + src.step[1] * j + src.elemSize1() * 2));
				if(pow(B,2.0)+pow(R,2.0)+pow(G,2.0) <= pow(thresh,2.0))
					memset(dst.data + dst.step[0] * row + dst.step[1] * j,maxVal,sizeof(unsigned char)*dst.channels());
				else 
					memset(dst.data + dst.step[0] * row + dst.step[1] * j,0,sizeof(unsigned char)*dst.channels());
			}
		}  
		break;  

	case cv::THRESH_TRUNC:  
		for (int row = 0; row < src.rows;row++) {  
			for (int j = 0; j < src.cols; j++)  
			{
				int B = (*(src.data + src.step[0] * row + src.step[1] * j + src.elemSize1() * 0));
				int R = (*(src.data + src.step[0] * row + src.step[1] * j + src.elemSize1() * 1));
				int G = (*(src.data + src.step[0] * row + src.step[1] * j + src.elemSize1() * 2));
				if(pow(B,2.0)+pow(R,2.0)+pow(G,2.0) > pow(thresh,2.0))
					memset(dst.data + dst.step[0] * row + dst.step[1] * j,thresh,sizeof(unsigned char)*dst.channels());
				else 
					memcpy(dst.data + dst.step[0] * row + dst.step[1] * j,src.data + src.step[0] * row + src.step[1] * j,sizeof(unsigned char)*dst.channels());
			}
		}  
		break;  

	case cv::THRESH_TOZERO: 
		for (int row = 0; row < src.rows;row++) {  
			for (int j = 0; j < src.cols; j++)  
			{
				int B = (*(src.data + src.step[0] * row + src.step[1] * j + src.elemSize1() * 0));
				int R = (*(src.data + src.step[0] * row + src.step[1] * j + src.elemSize1() * 1));
				int G = (*(src.data + src.step[0] * row + src.step[1] * j + src.elemSize1() * 2));
				if(pow(B,2.0)+pow(R,2.0)+pow(G,2.0) <= pow(thresh,2.0))
					memset(dst.data + dst.step[0] * row + dst.step[1] * j,0,sizeof(unsigned char)*dst.channels());
				else 
					memcpy(dst.data + dst.step[0] * row + dst.step[1] * j,src.data + src.step[0] * row + src.step[1] * j,sizeof(unsigned char)*dst.channels());
			}
		}    
		break;  

	case cv::THRESH_TOZERO_INV:  
		for (int row = 0; row < src.rows;row++) {  
			for (int j = 0; j < src.cols; j++)  
			{
				int B = (*(src.data + src.step[0] * row + src.step[1] * j + src.elemSize1() * 0));
				int R = (*(src.data + src.step[0] * row + src.step[1] * j + src.elemSize1() * 1));
				int G = (*(src.data + src.step[0] * row + src.step[1] * j + src.elemSize1() * 2));
				if(pow(B,2.0)+pow(R,2.0)+pow(G,2.0) > pow(thresh,2.0))
					memset(dst.data + dst.step[0] * row + dst.step[1] * j,0,sizeof(unsigned char)*dst.channels());
				else 
					memcpy(dst.data + dst.step[0] * row + dst.step[1] * j,src.data + src.step[0] * row + src.step[1] * j,sizeof(unsigned char)*dst.channels());
			}
		}   

		break;  
	default:  
		printf("BadArg");  
	}  

}

void absdiff(const cv::Mat &src1, const cv::Mat &src2, cv::Mat &dst)
{
	assert(src1.dims == 2 && src2.dims == 2 && \
		((src1.type() == CV_8UC1 && src2.type() == CV_8UC1)|| (src1.type() == CV_8UC3 && src2.type() == CV_8UC3)) &&\
		src1.depth() == CV_8U && src2.depth() == CV_8U);

	dst.create(src1.size(),src1.type());
	uchar *row_data1 = NULL, *row_data2 = NULL, *row_data3 = NULL;
	switch(dst.type())
	{
	case CV_8UC1:
		{
			for (int row = 0; row < src2.rows; row++)  //256
			{
				row_data1 = src1.data + src1.step[0]*row;
				row_data2 = src2.data + src2.step[0]*row;
				row_data3 = dst.data + dst.step[0]*row;
				for (int col = 0; col < src2.cols; col++)  //320
				{
					uchar *data1 = row_data1 + src1.step[1]*col;
					uchar *data2 = row_data2 + src2.step[1]*col;
					uchar *data3 = row_data3 + dst.step[1]*col;
					if(*data1 == 0)
						*data3 = 0;
					else
						*data3 = abs(*data2 - *data1);	
				}
			}
		}
		break;
	case CV_8UC3:
		{
			for (int row = 0; row < src1.rows;row++) //256
			{  
				row_data1 = src1.data + src1.step[0]*row;
				row_data2 = src2.data + src2.step[0]*row;
				row_data3 = dst.data + dst.step[0]*row;
				for (int col = 0; col < src1.cols; col++)  //320
				{
					uchar *data1 = row_data1 + src1.step[1]*col;
					uchar *data2 = row_data2 + src2.step[1]*col;
					uchar *data3 = row_data3 + dst.step[1]*col;
					if(data1[0] == 0 && data1[1] == 0 && data1[2] == 0)
					{
						data3[0] = 0; data3[1] = 0; data3[2] = 0;
					}
					else
					{
						data3[0] = abs(data2[0] - data1[0]); 
						data3[1] = abs(data2[1] - data1[1]); 
						data3[2] = abs(data2[2] - data1[2]);
					}
					//cout <<src2.rows<<" "<<src2.cols<<" "<< y <<","<<x<<"\t"<<B<<" "<<R<<" "<<G<<endl;
				}
			}
		}
		break;
	}
	//cv::waitKey();
}

// 彩色图像的直方图均衡化  mingtian
void equalizeHistColorImage(const cv::Mat src, cv::Mat &dst)  
{  
	assert(src.channels() == 3 && src.dims == 2 && src.depth() == CV_8U);

	cv::Mat src_YCrCb;
	cv::cvtColor(src, src_YCrCb, cv::COLOR_BGR2YCrCb);  

	vector<cv::Mat> channels;  
	cv::split(src_YCrCb, channels);  
	equalizeHist(channels[0], channels[0] ); 

	cv::merge(channels, dst);  
	cv::cvtColor(dst, dst, cv::COLOR_YCrCb2BGR);  
}  

//打印图像数据
void printImage(cv::Mat img, char *file)
{
	assert(img.dims == 2);

	if(file[0] != 0)
		freopen(file,"w",stdout);

	printf("%d\t(%4d,%4d)\n",img.channels(),img.rows,img.cols);

	uint8_t x[3];
	for(int row=0;row<img.rows;row++)
	{

		for(int col = 0;col<img.cols;col++)
		{
			if(col%30 == 0)printf("\n");
			for(int chan = 0;chan <img.channels();chan++)				
				x[chan] = (*(img.data + img.step[0] * row + img.step[1] * col + img.elemSize1() * chan));
			if(img.channels() == 1)printf("(%3d)",x[0]);
			else printf("(%3d,%3d,%3d)",x[0],x[1],x[2]);

		}
		printf("\n");
	}
	printf("\n===============================================================");
}

//提取特征点
int getKeyPointsBySift(const cv::Mat &src, KeyPointVector &keyPoint)
{
	cv::SiftFeatureDetector siftDetector;
	siftDetector.detect(src, keyPoint);
	//cv::Mat res;   
	//int drawmode = cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS;  
	//drawKeypoints(src,keyPoint,res,cv::Scalar::all(-1),drawmode);//在内存中画出特征点    
	//cv::imshow("KeyPoints of image1",res); 
	//cout<<"size of description of Img1: "<<keyPoint.size()<<endl;  
	return keyPoint.size();
}

//提取特征点的特征向量（128维）
int getPointsDescBySift(const cv::Mat &src, KeyPointVector &keyPoint, cv::Mat &pointsDesc)
{
	cv::SiftDescriptorExtractor siftDescriptor; 
	siftDescriptor.compute(src, keyPoint, pointsDesc); 
	return keyPoint.size();
}

//匹配特征点，主要计算两个特征点特征向量的欧式距离，距离小于某个阈值则认为匹配
int matchPointsDesc(const cv::Mat &pointsDesc1, const cv::Mat &pointsDesc2, DMatchVector &matchePoints)
{
	cv::BFMatcher matcher(cv::NORM_L2);    
	matcher.match(pointsDesc1, pointsDesc2, matchePoints); 
	sort(matchePoints.begin(), matchePoints.end()); //特征点排序  

	return matchePoints.size();
}

//获取最优匹配点对S
int getOptimalPointPairs(\
	vector<cv::DMatch> &matchePoints, const KeyPointVector &keyPoint1, const KeyPointVector &keyPoint2,\
	Point2fVector &points1, Point2fVector &points2, unsigned int N = 0)
{
	//cout<<"##"<<N<<" "<<matchePoints.size()<<endl;
	assert(N>=0 && N<=matchePoints.size());

	if(N == 0)N = matchePoints.size();
	if(!points1.empty())points1.clear();
	if(!points2.empty())points2.clear();

	for (size_t i = 0; i< N; i++)
	{
		points1.push_back(keyPoint1[matchePoints[i].queryIdx].pt);
		points2.push_back(keyPoint2[matchePoints[i].trainIdx].pt);
	}
	return N;
}

//利用基础矩阵剔除误匹配点
int optimizeMatcherByRansac( DMatchVector &matchePoints, \
	KeyPointVector &keyPoint1, KeyPointVector &keyPoint2, Point2fVector &points1, Point2fVector &points2, \
	const cv::Mat *mat1 = NULL, const cv::Mat *mat2 = NULL)
{
	vector<uchar> RansacStatus;
	cv::Mat Fundamental = findFundamentalMat(points1, points2, RansacStatus, cv::FM_RANSAC);

	KeyPointVector keyPoint01(keyPoint1), keyPoint02(keyPoint2);
	DMatchVector tmpMatchPoints(matchePoints);            //重新定义RR_keypoint 和RR_matches来存储新的关键点和匹配矩阵
	keyPoint1.clear();
	keyPoint2.clear();
	matchePoints.clear();
	int index=0;
	for (size_t i=0;i<RansacStatus.size();i++)
	{
		if (RansacStatus[i]!=0)
		{
			keyPoint1.push_back(keyPoint01[tmpMatchPoints[i].queryIdx]);
			keyPoint2.push_back(keyPoint02[tmpMatchPoints[i].trainIdx]);
			tmpMatchPoints[i].queryIdx=index;
			tmpMatchPoints[i].trainIdx=index;
			matchePoints.push_back(tmpMatchPoints[i]);
			index++;
		}
	}
	if(mat1 != NULL && mat2 != NULL)
	{
		cv::Mat img_show_matches;
		drawMatches(*mat1, keyPoint1, *mat2, keyPoint2, matchePoints, img_show_matches);
		sort(matchePoints.begin(), matchePoints.end()); //特征点排序  
		imshow("消除误匹配点后1",img_show_matches);
	}
	return matchePoints.size();
}

//sift算法
int getHomoMatBySift(const cv::Mat &src1, const cv::Mat &src2, cv::Mat &transMat)
{
	//提取特征点
	KeyPointVector keyPoint1, keyPoint2;  
	getKeyPointsBySift(src1, keyPoint1);
	getKeyPointsBySift(src2, keyPoint2);

	//提取特征点的特征向量（128维）
	cv::Mat imageDesc1,imageDesc2;//descriptor  
	getPointsDescBySift(src1,keyPoint1,imageDesc1);
	getPointsDescBySift(src2,keyPoint2,imageDesc2);    

	//匹配特征点，主要计算两个特征点特征向量的欧式距离，距离小于某个阈值则认为匹配
	vector<cv::DMatch> matchePoints;
	matchPointsDesc(imageDesc1, imageDesc2, matchePoints);

	//cv::Mat img_match;  
	//drawMatches(src1, keyPoint1, src2, keyPoint2, matchePoints, img_match);//,Scalar::all(-1),Scalar::all(-1),vector<char>(),drawmode);  
	//cout<<"number of matched points: "<< matchePoints.size() << endl; 

	//获取排在前N个的最优匹配特征点 
	Point2fVector points1, points2;
	getOptimalPointPairs(matchePoints, keyPoint1, keyPoint2, points1, points2, min<int>(20, matchePoints.size()));
	//cv::imshow( "source1_window", src_frame1 );
	//cv::imshow( "source2_window", src_frame2 );

#if (USET_SAUSAC)
	//利用基础矩阵剔除误匹配点
	if(optimizeMatcherByRansac(matchePoints, keyPoint1, keyPoint2, points1, points2/*, &src1, &src2*/)<=0)
		return -1;
	getOptimalPointPairs(matchePoints, keyPoint1, keyPoint2, points1, points2, min<int>(10, matchePoints.size()));
	//cv::destroyAllWindows();
#endif

	//获取图像1到图像2的投影映射矩阵，尺寸为3*3  
	transMat = cv::findHomography(points1, points2, CV_FM_RANSAC);
	//cv::waitKey();   
	return 0;
}

//获得目标直方图
int getObjectHist(const cv::Mat &src, const RectVector &objectRect, MatVector &hist)
{
	assert(src.data != NULL && src.channels() == 3);
	cv::Mat hsv, hue;
	cv::Mat mask;
	hist.resize(objectRect.size());

	cvtColor(src, hsv, cv::COLOR_BGR2HSV);
	inRange(hsv, Scalar(0, 0, 0),\
		Scalar(180, 256, 256), mask);
	int ch[] = {0, 0};
	hue.create(hsv.size(), hsv.depth());
	cv::mixChannels(&hsv, 1, &hue, 1, ch, 1);

	//计算H分量的直方图
	int hsize = 16;//将H分量的值量化到[0, 255]
	float hranges[] = {0,180};//H分量的取值范围是[0, 360]
	const float* phranges = hranges;
	for(int i=0; i<objectRect.size(); i++)
	{
		cv::Mat roi(hue, objectRect[i]);
		cv::Mat maskroi(mask, objectRect[i]);
		cv::calcHist(&roi, 1, 0, maskroi, hist[i], 1, &hsize, &phranges);
		//cv:calcHist(&roi, 1, 0, cv::Mat(), hist, &hsize, &phranges);
		cv::normalize(hist[i], hist[i], 0, 255, CV_MINMAX);
	}

	//画直方图
	/*cv::Mat histimg = Mat::zeros(200, 320, CV_8UC3);
	histimg = Scalar::all(0);
	int binW = histimg.cols / hsize;
	Mat buf(1, hsize, CV_8UC3);
	for( int i = 0; i < hsize; i++ )
	buf.at<Vec3b>(i) = Vec3b(saturate_cast<uchar>(i*180./hsize), 255, 255);
	cvtColor(buf, buf, CV_HSV2BGR);
	for( int i = 0; i < hsize; i++ )
	{
	int val = saturate_cast<int>(hist.at<float>(i)*histimg.rows/255);
	rectangle( histimg, Point(i*binW,histimg.rows),
	Point((i+1)*binW,histimg.rows - val),
	Scalar(buf.at<Vec3b>(i)), -1, 8 );
	}*/

	return 0;
}

//meanshift算法
int tracingByMeanshift(int &iImageIndex, RectVector &objectRect)
{
	float hranges[] = {0,180};//H分量的取值范围是[0, 360]
	const float* phranges = hranges;
	cv::Mat src, hsv, hue;
	cv::Mat backproj;
	MatVector hist;
	int ch[] = {0,0};
	if(readImage(iImageIndex, src, CV_LOAD_IMAGE_COLOR))return 0;
	getObjectHist( src, objectRect, hist);

	for(;;)
	{
		
		iImageIndex +=VIDEO_FRAME_INTERVAL_TRACE ;
		if(readImage(iImageIndex, src, CV_LOAD_IMAGE_COLOR) == -1)return -1;
		cvtColor(src, hsv, cv::COLOR_RGB2HSV);
		cout<<"\n###### Image: "<< iImageIndex<<"st #####"<<" "<<endl;
		
		hue.create(hsv.size(), hsv.depth());

		cv::mixChannels(&hsv, 1, &hue, 1, ch, 1);

		vector<bool> isValid(hist.size(), false);
		cv::Mat tmp = src.clone();
		int iTracedObject = 0;//检测到的目标个数
		for(int i= 0;i<hist.size();i++)
		{
			cv::Rect tmpRect = objectRect[i];
			calcBackProject(&hue, 1, 0, hist[i], backproj, &phranges);
			/*RotatedRect trackBox = */
			if(meanShift(backproj, tmpRect,\
				TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ))==0)continue;
			else if(tmpRect.area() < CONTOUR_MAX_AERA && tmpRect.area() > CONTOUR_MIN_AERA)
			{
				objectRect[i] = tmpRect;
				isValid[i] = true;
				cv::rectangle(tmp, tmpRect, cv::Scalar(0,0,255));
				iTracedObject++;
			}
		}
		waitKey(iDelay*5);
		cout << "目标数： "<<iTracedObject<<endl;
		for(int i=0;i<hist.size();i++)
			if(isValid[i])
				cout<<objectRect[i]<<endl;
		PAUSE;
		if(iTracedObject == 0)break;
		imshow("Result",tmp);

	}
	//destroyAllWindows();
	return 0;
}

int threeFrameDiff(int &iImageIndex, vector<cv::Rect> &objectRect)
{
	cv::Mat dst_frame;
	cv::Mat src_frame[MAX_FRAME];
	cv::Mat src_gray_frame[MAX_FRAME];
	int iIndex = 0;
	cv::Mat H12, H32;
	cv::Mat silh, mhi;
	cv::Size size;

	//加载第一张图片并均值滤波
	if(readImage(iImageIndex, src_frame[iIndex], CV_LOAD_IMAGE_COLOR) == -1)return -1;
	if(readImage(iImageIndex, src_gray_frame[iIndex%MAX_FRAME], CV_LOAD_IMAGE_GRAYSCALE) == -1)return -1;
	cv::blur(src_gray_frame[iIndex], src_gray_frame[iIndex], cv::Size(5,5));

#if (MAX_FRAME==3)
	//加载第二张图片并均值滤波
	iIndex ++; iImageIndex+=VIDEO_FRAME_INTERVAL_DETECTIVE;
	if(readImage(iImageIndex, src_frame[iIndex], CV_LOAD_IMAGE_COLOR) == -1)return -1;
	if(readImage(iImageIndex, src_gray_frame[iIndex%MAX_FRAME], CV_LOAD_IMAGE_GRAYSCALE) == -1)return -1;
	cv::blur(src_gray_frame[iIndex], src_gray_frame[iIndex], cv::Size(3,3));
#endif

	//获取图片大小
	size = src_gray_frame[iIndex].size();
	iIndex ++;
	while(true)
	{		
		objectRect.clear();
		double timestamp = clock() / 100.;
		//加载源图像
		iImageIndex+=VIDEO_FRAME_INTERVAL_DETECTIVE;
		if(readImage(iImageIndex, src_frame[iIndex%MAX_FRAME], CV_LOAD_IMAGE_COLOR) == -1)return -1;
		if(readImage(iImageIndex, src_gray_frame[iIndex%MAX_FRAME], CV_LOAD_IMAGE_GRAYSCALE) == -1)return -1;
		cout<<"\n###### Image: "<< iImageIndex <<"st\tin"<<iIndex<<" #####\t"<<endl;

		/*imshow("111",src_frame[iIndex%MAX_FRAME]);
		cv::GaussianBlur(src_frame[iIndex%MAX_FRAME], src_frame[iIndex%MAX_FRAME],  Size(13, 13),0);
		imshow("222",src_frame[iIndex%MAX_FRAME]);
*/
		//图像预处理
		cv::blur(src_gray_frame[iIndex%MAX_FRAME], src_gray_frame[iIndex%MAX_FRAME], cv::Size(3,3));
		cv::Mat tmp1, tmp2;

#if (USE_SIFT)
		//获得透视矩阵
#if (MAX_FRAME == 3)
		if(getHomoMatBySift(src_gray_frame[(iIndex-2)%MAX_FRAME], src_gray_frame[(iIndex-1)%MAX_FRAME], H12) != 0)
			continue;
		//变换图像
		cv::warpPerspective(src_gray_frame[(iIndex-2)%MAX_FRAME], tmp1, H12, src_gray_frame[(iIndex-1)%MAX_FRAME].size());
		absdiff(tmp1, src_gray_frame[(iIndex-1)%MAX_FRAME], tmp1);
#endif
		if(getHomoMatBySift(src_gray_frame[iIndex%MAX_FRAME], src_gray_frame[(iIndex-1)%MAX_FRAME], H32) != 0)
			continue;
		cv::warpPerspective(src_gray_frame[iIndex%MAX_FRAME], tmp2, H32, src_gray_frame[(iIndex-1)%MAX_FRAME].size());
		absdiff(tmp2, src_gray_frame[(iIndex-1)%MAX_FRAME], tmp2);
		//取重叠部位
		//printImage(src_gray_frame1,"C:\\Users\\liupengfei\\Desktop\\1.txt");
		//printImage(src_gray_frame2,"C:\\Users\\liupengfei\\Desktop\\2.txt");
#else	
#if (MAX_FRAME == 3)
		absdiff(src_gray_frame[(iIndex)%MAX_FRAME], src_gray_frame[(iIndex-1)%MAX_FRAME], tmp1);
#endif //end (MAX_FRAME == 3)
		absdiff(src_gray_frame[iIndex%MAX_FRAME], src_gray_frame[(iIndex-1)%MAX_FRAME], tmp2);
#endif	//end (USE_SIFT)

		//二值化操作
#if (MAX_FRAME == 3)
		cv::threshold(tmp1,tmp1,25,255,THRESH_BINARY);
		cv::threshold(tmp2,tmp2,25,255,THRESH_BINARY);
		cv::bitwise_and(tmp1,tmp2,silh);
#else
		cv::threshold(tmp2,silh,25,255,THRESH_BINARY);
#endif
		//更新运动
		//imshow("silh",silh);
		if(!mhi.data)
			mhi = Mat::zeros(size, CV_32FC1);

		if(!dst_frame.data)
			dst_frame = Mat::zeros(size, CV_8UC1);
		updateMotionHistory(silh, mhi, timestamp, MHI_DURATION);

#if (PYR_DOWN_UP)
		cv::Mat pyr;
		pyr = (255. / MHI_DURATION) * mhi + ((MHI_DURATION - timestamp)*255. / MHI_DURATION);
		cv::blur(pyr, pyr, cv::Size(3,3));
		pyrDown(pyr, pyr);
		//cvPyrDown(dst, pyr, CV_GAUSSIAN_5x5);// 向下采样，去掉噪声，图像是原图像的四分之一
		dilate(pyr, pyr, cv::Mat(),cv::Point(-1,-1), 1); // 做膨胀操作，消除目标的不连续空洞   
		pyrUp(pyr, pyr);// 向上采样，恢复图像，图像是原图像的四倍 
		cv::threshold(pyr,pyr,1,255,THRESH_BINARY);
		pyr.convertTo(dst_frame,CV_8UC1);
#else
		//cv::erode(mhi, dst_frame, cv::Mat(),cv::Point(-1,-1), 1);
		cv::dilate(mhi, dst_frame, cv::Mat(),cv::Point(-1,-1), 3);
		cv::erode(dst_frame, dst_frame, cv::Mat(),cv::Point(-1,-1), 2);
		//cv::dilate(mhi, dst_frame, cv::Mat(),cv::Point(-1,-1), 3);
		
		cv::threshold(dst_frame,dst_frame,1,255,THRESH_BINARY);
		dst_frame.convertTo(dst_frame,CV_8UC1);
#endif
		imshow("mhi",mhi);
		//轮廓   
		cv::imshow( "result_window", dst_frame );
#if (MAX_FRAME == 3)
		cv::Mat tmp = src_frame[(iIndex-1)%MAX_FRAME].clone();
#else
		cv::Mat tmp = src_frame[(iIndex)%MAX_FRAME].clone();
#endif
		std::vector<std::vector<cv::Point> > contours;  
		cv::findContours(dst_frame, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);    
		std::vector<std::vector<cv::Point> >::const_iterator itContours= contours.begin();
		for ( ; itContours!=contours.end(); ++itContours) 
		{
			cv::Rect rect = cv::boundingRect(*itContours);
			if(rect.height * rect.width < CONTOUR_MAX_AERA && rect.height * rect.width > CONTOUR_MIN_AERA)
			{
				cv::rectangle(tmp, rect, cv::Scalar(0,0,255));
				objectRect.push_back(rect);
			}
		}
		cout << "目标数： "<<objectRect.size()<<endl;
		for(int i=0;i<objectRect.size();i++)
			cout<<objectRect[i]<<endl;
		//显示结果
		cv::imshow( "Result", tmp );
		//cv::imshow( "source1_window", src_gray_frame[(iIndex-2)%MAX_FRAME] );
		//cv::imshow( "source2_window", src_gray_frame[(iIndex-1)%MAX_FRAME] );
		//cv::imshow( "source3_window", src_gray_frame[iIndex%MAX_FRAME] );
		cv::waitKey(iDelay);
		
#if (USE_TRACERING)
		if(!objectRect.empty())
		{
			PAUSE;
			return 0;
		}
#endif
		PAUSE;
		iIndex ++;

	}
	//destroyAllWindows();
	return 0;
}

int threeFrameDiff1(int &iImageIndex, vector<cv::Rect> &rect)
{
	cv::Mat dst_frame;
	cv::Mat src_frame[MAX_FRAME];
	cv::Mat src_gray_frame[MAX_FRAME];

	char imgName[100];
	cv::Mat H13, H23;
	int iIndex = 0;

	//加载第一张图片
	if(readImage(iImageIndex, src_frame[iIndex], CV_LOAD_IMAGE_COLOR) == -1)return -1;
	if(readImage(iImageIndex, src_gray_frame[iIndex], IMREAD_GRAYSCALE) == -1)return -1;
	cv::blur(src_gray_frame[iIndex], src_gray_frame[iIndex], cv::Size(3,3));


	//加载第二张图片
	iImageIndex ++; iIndex += VIDEO_FRAME_INTERVAL_DETECTIVE;
	if(readImage(iImageIndex, src_frame[iIndex], CV_LOAD_IMAGE_COLOR) == -1)return -1;
	if(readImage(iImageIndex, src_gray_frame[iIndex], IMREAD_GRAYSCALE) == -1)return -1;
	cv::blur(src_gray_frame[iIndex], src_gray_frame[iIndex], cv::Size(3,3));

	iIndex ++;
	while(true)
	{			
		//加载源图像
		iImageIndex += VIDEO_FRAME_INTERVAL_DETECTIVE;
		if(readImage(iImageIndex, src_frame[iIndex%MAX_FRAME], CV_LOAD_IMAGE_COLOR) == -1)return -1;
		if(readImage(iImageIndex, src_gray_frame[iIndex%MAX_FRAME], IMREAD_GRAYSCALE) == -1)return -1;
	
		//图像预处理
		cv::blur(src_gray_frame[iIndex%MAX_FRAME], src_gray_frame[iIndex%MAX_FRAME], cv::Size(3,3));
		
		cv::Mat tmp1 = src_gray_frame[(iIndex-2)%MAX_FRAME].clone(), tmp2 = src_gray_frame[(iIndex-1)%MAX_FRAME].clone();
#if (USE_SIFT)
		//获得透视矩阵
		if(getHomoMatBySift(src_gray_frame[(iIndex-2)%MAX_FRAME], src_gray_frame[iIndex%MAX_FRAME], H13) != 0)
			continue;
		if(getHomoMatBySift(src_gray_frame[(iIndex-1)%MAX_FRAME], src_gray_frame[iIndex%MAX_FRAME], H23) != 0)
			continue;
		cout<<"Image "<< iImageIndex <<"\t\n"<< H13<<" ... "<<endl;

		//变换图像
		cv::warpPerspective(src_gray_frame[(iIndex-2)%MAX_FRAME], tmp1, H13, src_gray_frame[iIndex%MAX_FRAME].size());
		cv::warpPerspective(src_gray_frame[(iIndex-1)%MAX_FRAME], tmp2, H23, src_gray_frame[iIndex%MAX_FRAME].size());		
#endif

		//printImage(src_gray_frame1,"C:\\Users\\liupengfei\\Desktop\\1.txt");
		//printImage(src_gray_frame2,"C:\\Users\\liupengfei\\Desktop\\2.txt");
		absdiff(tmp1, src_gray_frame[iIndex%MAX_FRAME], tmp1);
		absdiff(tmp2, src_gray_frame[iIndex%MAX_FRAME], tmp2);
		cv::threshold(tmp1,tmp1,20,255,CV_THRESH_BINARY);
		cv::threshold(tmp2,tmp2,20,255,CV_THRESH_BINARY);
		bitwise_and(tmp1, tmp2, dst_frame);

		//cv::imshow("absdiff",dst_frame);
		//cv::threshold(dst_frame,dst_frame,20,255,CV_THRESH_BINARY);
		cv::erode(dst_frame, dst_frame, cv::Mat(),cv::Point(-1,-1),3);
		//cv::morphologyEx(dst_frame, dst_frame, cv::MORPH_OPEN, cv::Mat(3,3,CV_8U,cv::Scalar(1)), cv::Point(-1,-1), 2);
		cv::dilate(dst_frame, dst_frame, cv::Mat(),cv::Point(-1,-1),3);


		//轮廓   
		/*cv::Mat tmp = dst_frame.clone();
		std::vector<std::vector<cv::Point> > contours;   
		cv::findContours(tmp, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);    
		std::vector<std::vector<cv::Point> >::const_iterator itContours= contours.begin();
		for ( ; itContours!=contours.end(); ++itContours) 
		{
		cv::Rect rect = cv::boundingRect(*itContours);
		cv::rectangle(src_frame[iIndex%MAX_FRAME], rect, cv::Scalar(0,0,255));
		}*/

		//显示结果
		cv::imshow( "source1_window", src_gray_frame[(iIndex-2)%MAX_FRAME] );
		cv::imshow( "source2_window", src_gray_frame[(iIndex-1)%MAX_FRAME] );
		cv::imshow( "source3_window", src_gray_frame[iIndex%MAX_FRAME] );
		cv::imshow( "result_window", dst_frame );
		cv::imshow( "src_frame3", src_frame[iIndex%MAX_FRAME] );
		cv::waitKey(iDelay);

		iIndex ++;

	}
	//等待用户按键退出程序
	cv::waitKey(0);
	return 0;
}

int main(int argc, char* argv[])
{
	int iImageIndex = 1;
	vector<cv::Rect> rect;
	while(true)
	{ 
		cout<<"=====================================================\n";
		cout << "开始检测目标 ..." << endl;
		if(threeFrameDiff(iImageIndex, rect) == -1) break;
		cout << "已经检测到目标, 开始跟踪 ..." << endl;
		if(tracingByMeanshift(iImageIndex, rect) == -1)break;
	}
	cout << "已结束 ..." << endl;
	getchar();
	return 0;
}