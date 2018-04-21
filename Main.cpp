#include <cv.h>
#include <highgui.h>
#include <vector>
#include <iostream>
#include "IntelligentScissors.h"

using namespace std;
using namespace cv;

#define Const(a,b) (int)(a##e##b)

void MouseCallbackFunc(int m_event, int x, int y, int flags, void *param)
{
	IntelligentScissors* intelligentScissors = (IntelligentScissors*)param;
	intelligentScissors->MouseCallbackFunc(m_event, x, y, flags, NULL);
}

int main(){

	Mat img = imread("0.jpg" );
	
	if (!img.data){
		cout << "Í¼Æ¬³ö´í\n";
		return -1;
	}

	IntelligentScissors intelligentScissors(img);

	const char* name = "IntelligentScissors";
	namedWindow(name, WINDOW_AUTOSIZE);

	setMouseCallback(name, MouseCallbackFunc, &intelligentScissors);
	imshow(name, img);

	waitKey();

	cvDestroyAllWindows();

	return 0;
}