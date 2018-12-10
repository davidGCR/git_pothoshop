#ifndef PROCESSING_H
#define PROCESSING_H

#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>

using namespace cv;
using namespace std;

class processing
{
public:
    processing();
    static void imhist(Mat image, int histogram[]);
    static void cumhist(int histogram[], int cumhistogram[]);
    static void histDisplay(int histogram[], const char* name);
    static void equalizarHistograma(string);
    static void bilinearTransform(Mat image,Mat* respt,Mat* inv,Rect, vector <Point>,float*,float*);
    static void transform(Mat coeff, int* x, int* y,float*,float*);
    static void inv_transform(Mat coeff, float* x, float* y,float*,float*);
    static void interpolate(float *px, float *py, float* u, float* v);
    static void applyFilter(Mat* image, Mat filter);
    static void insertionSort(int window[]);
    static void convolute(Mat src,Mat dst, Mat kernel);

};

#endif // PROCESSING_H
