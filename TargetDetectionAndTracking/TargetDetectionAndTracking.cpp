// TargetDetectionAndTracking.cpp : 定义控制台应用程序的入口点。

#include <stdio.h>
#include <iostream>
#include <queue>
#include <stdint.h>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <highgui/highgui.hpp>
#include <features2d/features2d.hpp>
#include <nonfree/nonfree.hpp>
#include <vector>

using namespace std;
//#define MAX(a,b) ((a)>(b)?(a):(b))
//#define MIN(a,b) ((a)<(b)?(a):(b))
#define VIDEO_FRAME_INTERVAL 3
typedef std::vector<cv::KeyPoint> KeyPointVector;
typedef std::vector<cv::Point2f> Point2fVector;
typedef std::vector<cv::DMatch> DMatchVector;


char g_ImgName[100] = "C:\\Users\\liupengfei\\Desktop\\data(4.23)\\data(4.23)\\data1\\imgs\\";
int iDelay = 200;

void equalizeHistColorImage(cv::Mat src, cv::Mat &dst);
void printImage(cv::Mat image, char *file);
void threshold(const cv::Mat &src, cv::Mat &dst, double thresh, double maxVal, int thresholdType);
void absdiff(const cv::Mat &src1, const cv::Mat &src2, cv::Mat &dst);
cv::Point2f getTransformPoint(const cv::Point2f originalPoint, const cv::Mat &transformMaxtri);

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
	cout<<"##"<<N<<" "<<matchePoints.size()<<endl;
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

// sift算法
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
	cout<<"number of matched points: "<< matchePoints.size() << endl; 

	//获取排在前N个的最优匹配特征点 
	Point2fVector points1, points2;
	getOptimalPointPairs(matchePoints, keyPoint1, keyPoint2, points1, points2, min<int>(20, matchePoints.size()));
	//cv::imshow( "source1_window", src_frame1 );
	//cv::imshow( "source2_window", src_frame2 );
	//利用基础矩阵剔除误匹配点
	if(optimizeMatcherByRansac(matchePoints, keyPoint1, keyPoint2, points1, points2, &src1, &src2)<=0)
		return -1;
	getOptimalPointPairs(matchePoints, keyPoint1, keyPoint2, points1, points2, min<int>(10, matchePoints.size()));
	//cv::destroyAllWindows();

	//获取图像1到图像2的投影映射矩阵，尺寸为3*3  
	transMat = cv::findHomography(points1, points2, CV_FM_RANSAC);
	//cv::waitKey();   
	return 0;
}



//计算原始图像点位在经过矩阵变换后在目标图像上对应位置  
cv::Point2f getTransformPoint(const cv::Point2f originalPoint, const cv::Mat &transformMaxtri)
{
	cv::Mat originelP, targetP;
	originelP = (cv::Mat_<double>(3, 1) << originalPoint.x, originalPoint.y, 1.0);
	targetP = transformMaxtri*originelP;
	float x = targetP.at<double>(0, 0) / targetP.at<double>(2, 0);
	float y = targetP.at<double>(1, 0) / targetP.at<double>(2, 0);
	return cv::Point2f(x, y);
}

int threeFrameDiff()
{
	cv::Mat dst_frame;
	cv::Mat src_frame[3];
	cv::Mat src_gray_frame[3];

	char imgName[100];
	int iIndex = 0;
	int iImageIndex = 0;
	cv::Mat H12, H32;

	//加载第一张图片
	sprintf(imgName,"%simg%05d.jpg",g_ImgName,iImageIndex*VIDEO_FRAME_INTERVAL + 1);
	src_frame[iIndex] = cv::imread(imgName, CV_LOAD_IMAGE_COLOR );
	src_gray_frame[iIndex] = cv::imread(imgName, cv::IMREAD_GRAYSCALE);
	if( !src_gray_frame[iIndex].data)
		return -1;
	cv::blur(src_gray_frame[iIndex], src_gray_frame[iIndex], cv::Size(5,5));
	iIndex ++; iImageIndex++;
	sprintf(imgName,"%simg%05d.jpg",g_ImgName,iImageIndex*VIDEO_FRAME_INTERVAL + 1);
	src_frame[iIndex] = cv::imread(imgName, CV_LOAD_IMAGE_COLOR );
	src_gray_frame[iIndex] = cv::imread(imgName, cv::IMREAD_GRAYSCALE);
	if(!src_gray_frame[iIndex].data)
	{
		cout << "Data Error! " << "Total "<<iIndex*VIDEO_FRAME_INTERVAL + 1<<"! "<<endl;
		return -1;
	}
	cv::blur(src_gray_frame[iIndex], src_gray_frame[iIndex], cv::Size(5,5));
	iIndex ++;
	while(true)
	{			
		//加载源图像
		iImageIndex++;
		memset(imgName, 0, sizeof(imgName));
		sprintf(imgName,"%simg%05d.jpg",g_ImgName, iImageIndex*VIDEO_FRAME_INTERVAL + 1);
		src_frame[iIndex%3] = cv::imread(imgName, CV_LOAD_IMAGE_COLOR );
		src_gray_frame[iIndex%3] = cv::imread(imgName, cv::IMREAD_GRAYSCALE);
		if( !src_gray_frame[iIndex%3].data )
		{
			cout << "Data finished! " << "Total "<< iImageIndex*VIDEO_FRAME_INTERVAL + 1 << "! "<<endl;
			break;
		}
		//图像预处理
		//cvtColor(src_frame2, src_gray_frame2,cv::COLOR_BGR2GRAY);
		cv::blur(src_gray_frame[iIndex%3], src_gray_frame[iIndex%3], cv::Size(5,5));

		//获得透视矩阵
		if(getHomoMatBySift(src_gray_frame[(iIndex-2)%3], src_gray_frame[(iIndex-1)%3], H12) != 0)
			continue;
		if(getHomoMatBySift(src_gray_frame[iIndex%3], src_gray_frame[(iIndex-1)%3], H32) != 0)
			continue;
		cout<<"###### Image "<< iImageIndex*VIDEO_FRAME_INTERVAL + 1 <<"\t"<<iIndex<<" #####\n\t"<< H12<<" "<<endl;

		//变换图像
		cv::Mat tmp1, tmp2;
		cv::warpPerspective(src_gray_frame[(iIndex-2)%3], tmp1, H12, src_gray_frame[(iIndex-1)%3].size());
		cv::warpPerspective(src_gray_frame[iIndex%3], tmp2, H32, src_gray_frame[(iIndex-1)%3].size());
		//取重叠部位
		//printImage(src_gray_frame1,"C:\\Users\\liupengfei\\Desktop\\1.txt");
		//printImage(src_gray_frame2,"C:\\Users\\liupengfei\\Desktop\\2.txt");
		
		absdiff(tmp1,src_gray_frame[(iIndex-1)%3], tmp1);
		absdiff(tmp2, src_gray_frame[(iIndex-1)%3], tmp2);
		bitwise_and(tmp1, tmp2, dst_frame);

		cv::imshow("absdiff",dst_frame);
		cv::threshold(dst_frame,dst_frame,20,255,CV_THRESH_BINARY);
		//cv::erode(dst_frame, dst_frame, cv::Mat());
		cv::morphologyEx(dst_frame, dst_frame, cv::MORPH_CLOSE, cv::Mat(3,3,CV_8U,cv::Scalar(1)), cv::Point(-1,-1), 1);
		//printImage(dst_frame,"C:\\Users\\liupengfei\\Desktop\\3.txt");

		//轮廓   
		cv::Mat tmp = dst_frame.clone();
		std::vector<std::vector<cv::Point> > contours;   
		cv::findContours(tmp, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);    
		std::vector<std::vector<cv::Point> >::const_iterator itContours= contours.begin();
		for ( ; itContours!=contours.end(); ++itContours) 
		{
			cv::Rect rect = cv::boundingRect(*itContours);
			cv::rectangle(src_frame[(iIndex-1)%3], rect, cv::Scalar(0,0,255));
		}

		//显示结果
		cv::imshow( "source1_window", src_gray_frame[(iIndex-2)%3] );
		cv::imshow( "source2_window", src_gray_frame[(iIndex-1)%3] );
		cv::imshow( "source3_window", src_gray_frame[iIndex%3] );
		cv::imshow( "result_window", dst_frame );
		cv::imshow( "src_frame3", src_frame[(iIndex-1)%3] );

		cv::waitKey(iDelay);

		iIndex ++;

	}
	//等待用户按键退出程序
	cv::waitKey(0);
	return 0;
}

int threeFrameDiff1()
{
	cv::Mat dst_frame;
	cv::Mat src_frame1, src_frame2, src_frame3;
	cv::Mat src_gray_frame1, src_gray_frame2, src_gray_frame3;

	char imgName[100];
	int iIndex = 0;
	cv::Mat H13, H23;

	//加载第一张图片
	sprintf(imgName,"%simg%05d.jpg",g_ImgName,iIndex*VIDEO_FRAME_INTERVAL + 1);
	src_frame1 = cv::imread(imgName, CV_LOAD_IMAGE_COLOR );
	src_gray_frame1 = cv::imread(imgName, cv::IMREAD_GRAYSCALE);
	iIndex ++;
	sprintf(imgName,"%simg%05d.jpg",g_ImgName,iIndex*VIDEO_FRAME_INTERVAL + 1);
	src_frame2 = cv::imread(imgName, CV_LOAD_IMAGE_COLOR );
	src_gray_frame2 = cv::imread(imgName, cv::IMREAD_GRAYSCALE);
	if( !src_gray_frame1.data || !src_gray_frame2.data)
	{
		cout << "Data Error! " << "Total "<<iIndex*VIDEO_FRAME_INTERVAL + 1<<"! "<<endl;
		return -1;
	}
	cv::blur(src_gray_frame1, src_gray_frame1, cv::Size(3,3));
	cv::blur(src_gray_frame2, src_gray_frame2, cv::Size(3,3));
	while(true)
	{			
		//加载源图像
		iIndex ++;
		memset(imgName, 0, sizeof(imgName));
		sprintf(imgName,"%simg%05d.jpg",g_ImgName,iIndex*VIDEO_FRAME_INTERVAL + 1);
		src_frame3 = cv::imread(imgName, CV_LOAD_IMAGE_COLOR );
		src_gray_frame3 = cv::imread(imgName, cv::IMREAD_GRAYSCALE);
		if( !src_gray_frame3.data )
		{
			cout << "Data finished! " << "Total "<< iIndex*VIDEO_FRAME_INTERVAL + 1 << "! "<<endl;
			break;
		}
		//图像预处理
		cv::blur(src_gray_frame3, src_gray_frame3, cv::Size(3,3));

		//获得透视矩阵
		if(getHomoMatBySift(src_gray_frame1, src_gray_frame3, H13) != 0)
			continue;
		if(getHomoMatBySift(src_gray_frame2, src_gray_frame3, H23) != 0)
			continue;
		cout<<"Image "<< iIndex*VIDEO_FRAME_INTERVAL + 1 <<"\t\n"<< H13<<" ... "<<endl;

		//变换图像
		cv::warpPerspective(src_gray_frame1, src_gray_frame1, H13, src_gray_frame3.size());
		cv::warpPerspective(src_gray_frame2, src_gray_frame2, H23, src_gray_frame3.size());
		//取重叠部位
		//printImage(src_gray_frame1,"C:\\Users\\liupengfei\\Desktop\\1.txt");
		//printImage(src_gray_frame2,"C:\\Users\\liupengfei\\Desktop\\2.txt");
		cv::Mat tmp1, tmp2;
		absdiff(src_gray_frame1, src_gray_frame3, tmp1);
		absdiff(src_gray_frame2, src_gray_frame3, tmp2);
		bitwise_and(tmp1, tmp2, dst_frame);

		cv::imshow("absdiff",dst_frame);
		cv::threshold(dst_frame,dst_frame,20,255,CV_THRESH_BINARY);
		//cv::erode(dst_frame, dst_frame, cv::Mat());
		//cv::morphologyEx(dst_frame, dst_frame, cv::MORPH_OPEN, cv::Mat(3,3,CV_8U,cv::Scalar(1)), cv::Point(-1,-1), 1);
		//printImage(dst_frame,"C:\\Users\\liupengfei\\Desktop\\3.txt");

		//轮廓   
		cv::Mat tmp = dst_frame.clone();
		std::vector<std::vector<cv::Point> > contours;   
		cv::findContours(tmp, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);    
		std::vector<std::vector<cv::Point> >::const_iterator itContours= contours.begin();
		for ( ; itContours!=contours.end(); ++itContours) 
		{
			cv::Rect rect = cv::boundingRect(*itContours);
			cv::rectangle(src_frame3, rect, cv::Scalar(0,0,255));
		}

		//显示结果
		cv::imshow( "source1_window", src_gray_frame1 );
		cv::imshow( "source2_window", src_gray_frame2 );
		cv::imshow( "source3_window", src_gray_frame3 );
		cv::imshow( "result_window", dst_frame );
		cv::imshow( "src_frame3", src_frame3 );
		cv::waitKey(iDelay);

		src_frame1 = src_frame2.clone();
		src_gray_frame1 = src_gray_frame2.clone();
		src_frame2 = src_frame3.clone();
		src_gray_frame2 = src_gray_frame3;

	}
	//等待用户按键退出程序
	cv::waitKey(0);
}

int main(int argc, char* argv[])
{
	threeFrameDiff();
	
	return 0;
}