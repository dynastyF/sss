#pragma once
#include <cv.h>
#include<opencv2/opencv.hpp>
#include "Vector2.h"

using namespace cv;


class IntelligentScissors{
public:
	IntelligentScissors(Mat& image);
	~IntelligentScissors();
	void DP(int sx, int sy);//寻路函数						
	void MouseCallbackFunc(int m_event, int x, int y, int flags, void *param);//鼠标回调事件

private:
	Mat& img;
	Mat target;
	uchar* fZ;// Laplacian Zero-Crossing //拉普拉斯交叉零点
	float* fG;// Gradient Magnitude  //拉普拉斯交叉零点
	Vector2<short>* D;// D`(p)
	float* Dlink; //D(link)
	int* paths; //记录路径
	float* ILC;  //图像区域耗费值(Image Local Cost) 是图像边界提取的依据
	bool isFirst; //记录鼠标回调函数调用DP

	void init();
	void init_fZ();
	void init_fG_D();
	void init_ILC();

	float computefD(int px, int py, int qx, int qy);
	float computeILC(int px, int py, int qx, int qy);


	uchar get_target_pixel(int x, int y) //
	{
		if (x < 0) 
			x = 0;
		if (x >= target.cols) 
			x = target.cols - 1;
		if (y < 0) 
			y = 0;
		if (y >= target.rows) 
			y = target.rows - 1;
		return target.data[x + y*target.cols];
	}


};
