#include<iostream>
#include <fstream>
#include <string>
#include<stdlib.h>
#include<opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;
//
//Mat map1 = imread("H:\\RM培训\\开源\\gcampus-radar-station-master\\radar-station\\map.jpg");
//Point2f mouse[4];
//int i = 0;
//Mat for_change_distort;// = imread("H:\\robomaster\\rader\\test4.jpg");
//Mat M1, dst;
//int main()
//{
//	VideoCapture cap;
//	cap.open("H:/test/temp17.avi");
//	//VideoCapture cap;
//	//cap.open(1);
//	if (!cap.isOpened())
//	{
//		std::cout << "video not open." << std::endl;
//		return 1;
//	}
//
//		Mat for_change_distort;
//	do
//	{
//		cap >> for_change_distort;
//		if (for_change_distort.empty())
//		{
//			cout << "emp" << endl;
//			break;
//		}
//		//cap.read(for_change_distort);
//		namedWindow("ss");
//		imshow("ss", for_change_distort);
//		int chKey = waitKey(1);
//
//		if (chKey == 27) {
//			break;
//		}
//	} while (true);
//	cap.release();
//	return 0;
//	
//}
	////获取当前视频帧率
	//double rate = cap.get(CV_CAP_PROP_FPS);
	////当前视频帧
	////cv::Mat frame;
	//cap.set(CAP_PROP_POS_FRAMES, 10);//fixed cap for frame 10 固定cap为第10帧图片，每10帧操作一次


int test02() {
	VideoCapture cam(1);
	if (!cam.isOpened())
	{
		cout << "cam open failed!" << endl;
		getchar();
		return -1;
	}
	cout << "cam open success!" << endl;
	namedWindow("cam");
	Mat img;
	VideoWriter vw;
	int fps = cam.get(CAP_PROP_FPS);  //获取摄像机帧率
	if (fps <= 0)fps = 25;
	//创建视频文件
	vw.open("C:\\Users\\75741\\Desktop\\1-1\\out2.avi", //路径
		VideoWriter::fourcc('M', 'J', 'P', 'G'), //编码格式
		fps, //帧率
		Size(cam.get(CAP_PROP_FRAME_WIDTH),
			cam.get(CAP_PROP_FRAME_HEIGHT))  //尺寸
	);
	if (!vw.isOpened())
	{
		cout << "VideoWriter open failed!" << endl;
		getchar();
		return -1;
	}
	cout << "VideoWriter open success!" << endl;

	for (;;)
	{
		cam.read(img);
		if (img.empty())break;
		imshow("cam", img);
		//写入视频文件
		vw.write(img);
		if (waitKey(5) == 'q') break;
	}

	waitKey(0);
	return 0;
}
void test01() {
	VideoCapture cap;
	cap.open("C:\\Users\\75741\\Desktop\\1-1\\out1.avi"); //打开视频,以上两句等价于VideoCapture cap("E://01.avi");

						   //cap.open("http://www.laganiere.name/bike.avi");//也可以直接从网页中获取图片，前提是网页有视频，以及网速够快
	if (!cap.isOpened())//如果视频不能正常打开则返回
		return;
	Mat frame;
	while (1)
	{
		cap >> frame;//等价于cap.read(frame);
		if (frame.empty())//如果某帧为空则退出循环
			break;
		imshow("video", frame);
		waitKey(20);//每帧延时20毫秒
	}
	cap.release();//释放资源
}
void test03()
{
	Mat src22 = imread("C:/Users/Peisen Zhao/Desktop/xml/001.png");
	cout << src22.channels();
	imwrite("C:/Users/Peisen Zhao/Desktop/xml/001.jpg", src22);
}
int main()
{
	test01();
	return 0;
}
//	return 0;
//}

//imshow("map1", map1);
//	void trans(Point2f point[4]);
//	void onMouseCallBack(int event, int x, int y, int flags, void* pUserData);
//	//pyrDown(for_change_distort, for_change_distort, Size(for_change_distort.cols / 2, for_change_distort.rows / 2));
//	imshow("windowName", for_change_distort);
//	cv::setMouseCallback("windowName", onMouseCallBack, reinterpret_cast<void*> (&for_change_distort));
//	char d = cv::waitKey();
//	if (d == 27) {
//		cv::destroyWindow("windowName");
//	}
//	warpPerspective(for_change_distort, dst, M1, Size(for_change_distort.cols, for_change_distort.rows), INTER_LINEAR);
//	imshow("trans", dst);
//	imshow("windowName", for_change_distort);
//	waitKey(0);
//	return 0;
//}
//
//void onMouseCallBack(int event, int x, int y, int flags, void* pUserData)
//{
//	void trans(Point2f point[4]);
//	float target_x[4], target_y[4];
//	target_x[3] = 0;//？？？
//
//	//创建保存像素值的3字节容器
//	cv::Vec3b pixel;
//	//空指针强制类型装换成图片指针pMat
//	cv::Mat* pMat = reinterpret_cast<cv::Mat*>(pUserData);
//
//	//鼠标左键按下时，返回坐标和RGB值
//	if (event == CV_EVENT_LBUTTONDOWN && i < 4)//用循环不好吗？
//	{
//		//获取像素值
//		pixel = pMat->at<cv::Vec3b>(y, x);//y是row，x是col
//									  //输出像素值的（R，G，B）
//									  //cv重载了<<运算符，可以输出Vec3b类型，但是按B,G,R输出
//		cout << "at(" << x << "," << y << ")-->pixel(B,G,R)=" << pixel << endl;
//		target_x[i] = x;
//		target_y[i] = y;
//		mouse[i] = cv::Point2f(target_x[i], target_y[i]);
//
//		i++;
//		cout << i << endl;
//	}
//
//	if (i == 4 && target_x[3] != 0) //为什么加一个条件？？？
//	{
//		trans(mouse);
//		i++;
//		//cout << target_x[3]<<endl;
//	}
//
//}
//void trans(Point2f point[4]) {
//	Mat dst_warp, dst_warpRotateScale, dst_warpTransformation, dst_warpFlip;
//
//	Point2f srcPoints[4];//Four points in the original picture 原图中的四点 ,一个包含三维点（x，y）的数组，其中x、y是浮点型数
//
//	Point2f dstPoints[4];//Four points in the target diagram 目标图中的四点  
//
//	//Writes the first annotated point to a file for subsequent reading  将第一次标注的点写入文件中，以便于后续读取
//	ofstream fw;
//	fw.open("a.txt", ios::trunc);
//	for (int i = 0; i < 4; i++) {
//		std::cout << point[i].x << point[i].y << endl;
//		fw << point[i].x << endl;
//		fw << point[i].y << endl;
//	}
//	fw.close();
//
//	//The four coordinates before mapping 映射前的四个坐标值
//	srcPoints[0] = point[0];
//
//	srcPoints[1] = point[1];
//
//	srcPoints[2] = point[2];
//
//	srcPoints[3] = point[3];
//
//	//The four coordinates after mapping 映射后的四个坐标值
//	//map1是映射矩阵？？？？？
//
//	dstPoints[0] = Point2f(0, 0);
//
//	dstPoints[1] = Point2f(0, map1.rows);
//
//	dstPoints[2] = Point2f(map1.cols, map1.rows);
//
//	dstPoints[3] = Point2f(map1.cols, 0);
//
//	//Calculate the perspective transformation matrix 由四个点对计算透视变换矩阵  
//
//	M1 = getPerspectiveTransform(srcPoints, dstPoints);
//
//	std::cout << M1 << endl;//调试用
//}