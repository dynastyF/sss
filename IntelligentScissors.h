#pragma once
#include <cv.h>
#include<opencv2/opencv.hpp>
#include "Vector2.h"

using namespace cv;


class IntelligentScissors{
public:
	IntelligentScissors(Mat& image);
	~IntelligentScissors();
	void DP(int sx, int sy);//Ѱ·����						
	void MouseCallbackFunc(int m_event, int x, int y, int flags, void *param);//���ص��¼�

private:
	Mat& img;
	Mat target;
	uchar* fZ;// Laplacian Zero-Crossing //������˹�������
	float* fG;// Gradient Magnitude  //������˹�������
	Vector2<short>* D;// D`(p)
	float* Dlink; //D(link)
	int* paths; //��¼·��
	float* ILC;  //ͼ������ķ�ֵ(Image Local Cost) ��ͼ��߽���ȡ������
	bool isFirst; //��¼���ص���������DP

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
