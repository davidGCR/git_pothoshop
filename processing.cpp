#include "processing.h"


processing::processing()
{

}

void processing::insertionSort(int window[])
{
    int temp, i , j;
    for(i = 0; i < 9; i++){
        temp = window[i];
        for(j = i-1; j >= 0 && temp < window[j]; j--){
            window[j+1] = window[j];
        }
        window[j+1] = temp;
    }
}

void processing::convolute(Mat src,Mat dst, Mat kernel)
{
//    Mat src, dst;

          // Load an image
//          src = imread(+"book.png", CV_LOAD_IMAGE_GRAYSCALE);

//          if( !src.data )
//          { return -1; }

          //create a sliding window of size 9
          int window[9];

//            dst = src.clone();
//            for(int y = 0; y < src.rows; y++)
//                for(int x = 0; x < src.cols; x++)
//                    dst.at<uchar>(y,x) = 0.0;

            for(int y = 1; y < src.rows - 1; y++){
                for(int x = 1; x < src.cols - 1; x++){

                    // Pick up window element

                    window[0] = src.at<Vec3b>(y - 1 ,x - 1)[0];
                    window[1] = src.at<Vec3b>(y, x - 1)[0];
                    window[2] = src.at<Vec3b>(y + 1, x - 1)[0];
                    window[3] = src.at<Vec3b>(y - 1, x)[0];
                    window[4] = src.at<Vec3b>(y, x)[0];
                    window[5] = src.at<Vec3b>(y + 1, x)[0];
                    window[6] = src.at<Vec3b>(y - 1, x + 1)[0];
                    window[7] = src.at<Vec3b>(y, x + 1)[0];
                    window[8] = src.at<Vec3b>(y + 1, x + 1)[0];

                    // sort the window to find median
                    insertionSort(window);

                    // assign the median to centered element of the matrix
                    dst.at<Vec3b>(y,x)[0] = window[4];

                    window[0] = src.at<Vec3b>(y - 1 ,x - 1)[1];
                    window[1] = src.at<Vec3b>(y, x - 1)[1];
                    window[2] = src.at<Vec3b>(y + 1, x - 1)[1];
                    window[3] = src.at<Vec3b>(y - 1, x)[1];
                    window[4] = src.at<Vec3b>(y, x)[1];
                    window[5] = src.at<Vec3b>(y + 1, x)[1];
                    window[6] = src.at<Vec3b>(y - 1, x + 1)[1];
                    window[7] = src.at<Vec3b>(y, x + 1)[1];
                    window[8] = src.at<Vec3b>(y + 1, x + 1)[1];

                    // sort the window to find median
                    insertionSort(window);

                    // assign the median to centered element of the matrix
                    dst.at<Vec3b>(y,x)[1] = window[4];

                    window[0] = src.at<Vec3b>(y - 1 ,x - 1)[2];
                    window[1] = src.at<Vec3b>(y, x - 1)[2];
                    window[2] = src.at<Vec3b>(y + 1, x - 1)[2];
                    window[3] = src.at<Vec3b>(y - 1, x)[2];
                    window[4] = src.at<Vec3b>(y, x)[2];
                    window[5] = src.at<Vec3b>(y + 1, x)[2];
                    window[6] = src.at<Vec3b>(y - 1, x + 1)[2];
                    window[7] = src.at<Vec3b>(y, x + 1)[2];
                    window[8] = src.at<Vec3b>(y + 1, x + 1)[2];

                    // sort the window to find median
                    insertionSort(window);

                    // assign the median to centered element of the matrix
                    dst.at<Vec3b>(y,x)[2] = window[4];
                }
            }

//            namedWindow("final");
//            imshow("final", dst);

//            namedWindow("initial");
//            imshow("initial", src);
}



void processing::transform(Mat coeff, int* x, int* y,float* u, float* v){
    float c11 = coeff.at<float>(0,0);
    float c12 = coeff.at<float>(0,1);
    float c13 = coeff.at<float>(0,2);
    float c14 = coeff.at<float>(0,3);
    float c21 = coeff.at<float>(0,4);
    float c22 = coeff.at<float>(0,5);
    float c23 = coeff.at<float>(0,6);
    float c24 = coeff.at<float>(0,7);
    *u = c11*(*x) + c12*(*y) + c13*(*x)*(*y) + c14;
    *v = c21*(*x)+ c22*(*y) + c23*(*x)*(*y) + c24;
}
void processing::inv_transform(Mat coeff, float* x, float* y,float* u, float* v){
    float c11 = coeff.at<float>(0,0);
    float c12 = coeff.at<float>(0,1);
    float c13 = coeff.at<float>(0,2);
    float c14 = coeff.at<float>(0,3);
    float c21 = coeff.at<float>(0,4);
    float c22 = coeff.at<float>(0,5);
    float c23 = coeff.at<float>(0,6);
    float c24 = coeff.at<float>(0,7);
    *x = c11*(*u) + c12*(*v) + c13*(*u)*(*v) + c14;
    *y = c21*(*u)+ c22*(*v) + c23*(*u)*(*v) + c24;
}
void processing::bilinearTransform(Mat image,Mat* img_result,Mat* img_inv,Rect rect,
                                   vector<Point> quadrilats, float*u,float*v){

    Size s = image.size();
    int rows = s.height;
    int cols = s.width;

//    float x1 = rect.x;
//    float y1= rect.y;
//    float x2 = rect.x+cols;
//    float y2= rect.y;
//    float x3 = rect.x+rows;
//    float y3= rect.y+cols;
//    float x4 = rect.x;
//    float y4= rect.y+rows;
    float x1 = 0;
    float y1= 0;
    float x2 = x1+cols;
    float y2= y1;
    float x3 = x2;
    float y3= y2+rows;
    float x4 = x1;
    float y4= y1+rows;

//    cout<<image<<endl;



    // the matrix of coefficients
    cv::Mat A = (cv::Mat_<float>(8,8) <<
                       x1, y1, x1*y1, 1,0, 0, 0, 0,
                       0,0,0,0,x1, y1, x1*y1, 1,
                         x2, y2, x2*y2, 1,0, 0, 0, 0,
                         0,0,0,0,x2, y2, x2*y2, 1,
                         x3, y3, x3*y3, 1,0, 0, 0, 0,
                         0,0,0,0,x3, y3, x3*y3, 1,
                         x4, y4, x4*y4, 1,0, 0, 0, 0,
                         0,0,0,0,x4, y4, x4*y4, 1

                 );

//   Vector b
    float m = x1*y1;
    cv::Mat b = (cv::Mat_<float>(8,1) <<
                 quadrilats[0].x,
                 quadrilats[0].y,
                 quadrilats[1].x,
                 quadrilats[1].y,
            quadrilats[2].x,
            quadrilats[2].y,
            quadrilats[3].x,
            quadrilats[3].y
                 );

    for(int i=0;i<quadrilats.size();i++){
        cout<<"("<<quadrilats[i].x<<","<<quadrilats[i].y<<")"<<endl;
    }

//  Calcular coeficientes de transformacion
    cv::Mat x;

    solve(A,b,x);
   cout << "transform: " << *u<<","<<*v<<endl;

    for(int j=0;j<rows;j++){
        for (int i=0;i<cols;i++){
            float ur,vr;
            float xinv, yinv;
            processing::transform(x,&i,&j,&ur,&vr);
            processing::inv_transform(x,&xinv,&yinv,&ur,&vr);
            Vec3b pixel = image.at<Vec3b>(i,j);
                    int b, g, r;
                    b = pixel[0];
                    g = pixel[1];
                    r = pixel[2];

                    (*img_result).at<Vec3b>(floor(vr),floor(ur))[0]=image.at<Vec3b>(j,i)[0];
                    (*img_result).at<Vec3b>(floor(vr),floor(ur))[1]=image.at<Vec3b>(j,i)[1];
                    (*img_result).at<Vec3b>(floor(vr),floor(ur))[2]=image.at<Vec3b>(j,i)[2];

//                    (*img_inv).at<Vec3b>(floor(yinv),floor(xinv))[0]=(*img_result).at<Vec3b>(floor(vr),floor(ur))[0];
//                    (*img_inv).at<Vec3b>(floor(yinv),floor(xinv))[1]=(*img_result).at<Vec3b>(floor(vr),floor(ur))[1];
//                    (*img_inv).at<Vec3b>(floor(yinv),floor(xinv))[2]=(*img_result).at<Vec3b>(floor(vr),floor(ur))[2];

            if(i==0&&j==0){
                cout<<"mapeo: ("<<ur<<","<<vr<<")"<<"pixel("<<b<<","<<g<<","<<r<<")+"<<"coordinate: "<<endl;
            }

         }
     }

    //INVERSA



}
void processing::interpolate(float *px, float *py, float* xf, float* yf){
    int i = floor(*px);
    int d = i+1;
    int s = floor(*py);
    int r = s+1;
    float a = (*px)-i;
    float b = (*py)-s;

}
 void processing::imhist(Mat image, int histogram[])
{

    // initialize all intensity values to 0
    for(int i = 0; i < 256; i++)
    {
        histogram[i] = 0;
    }

    // calculate the no of pixels for each intensity values
    for(int y = 0; y < image.rows; y++)
        for(int x = 0; x < image.cols; x++)
            histogram[(int)image.at<uchar>(y,x)]++;

}

 void processing::cumhist(int histogram[], int cumhistogram[])
{
    cumhistogram[0] = histogram[0];

    for(int i = 1; i < 256; i++)
    {
        cumhistogram[i] = histogram[i] + cumhistogram[i-1];
    }
}

 void processing::histDisplay(int histogram[], const char* name)
{
    int hist[256];
    for(int i = 0; i < 256; i++)
    {
        hist[i]=histogram[i];
    }
    // draw the histograms
    int hist_w = 512; int hist_h = 400;
    int bin_w = cvRound((double) hist_w/256);

    Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(255, 255, 255));

    // find the maximum intensity element from histogram
    int max = hist[0];
    for(int i = 1; i < 256; i++){
        if(max < hist[i]){
            max = hist[i];
        }
    }

    // normalize the histogram between 0 and histImage.rows

    for(int i = 0; i < 256; i++){
        hist[i] = ((double)hist[i]/max)*histImage.rows;
    }


    // draw the intensity line for histogram
    for(int i = 0; i < 256; i++)
    {
        line(histImage, Point(bin_w*(i), hist_h),
                              Point(bin_w*(i), hist_h - hist[i]),
             Scalar(0,0,0), 1, 8, 0);
    }

    // display histogram
//    namedWindow(name, CV_WINDOW_AUTOSIZE);
    namedWindow(name, WINDOW_AUTOSIZE);
    imshow(name, histImage);
}
 void processing::equalizarHistograma(string image_path){
     // Cargar imagen
//         Mat image = imread(image_path, CV_LOAD_IMAGE_GRAYSCALE);
     Mat image = imread(image_path);
     cvtColor(image, image, COLOR_BGR2GRAY);

         // Generar histograma
         int histogram[256];
         processing::imhist(image, histogram);

         // Imagen size
         int size = image.rows * image.cols;
         float alpha = 255.0/size;

         // probabilidad de cada intesidad
         float PrRk[256];
         for(int i = 0; i < 256; i++)
         {
             PrRk[i] = (double)histogram[i] / size;
         }

         // Acumular frecuencias de histograma
         int cumhistogram[256];
         processing::cumhist(histogram,cumhistogram );

         // Escalar histograma
         int Sk[256];
         for(int i = 0; i < 256; i++)
         {
             Sk[i] = cvRound((double)cumhistogram[i] * alpha);
         }


         // Generar hist. ecualizado
         float PsSk[256];
         for(int i = 0; i < 256; i++)
         {
             PsSk[i] = 0;
         }

         for(int i = 0; i < 256; i++)
         {
             PsSk[Sk[i]] += PrRk[i];
         }

         int final[256];
         for(int i = 0; i < 256; i++)
             final[i] = cvRound(PsSk[i]*255);


         // Generar imagen ecualizada
         Mat new_image = image.clone();

         for(int y = 0; y < image.rows; y++)
             for(int x = 0; x < image.cols; x++)
                 new_image.at<uchar>(y,x) = saturate_cast<uchar>(Sk[image.at<uchar>(y,x)]);

        // Display imagen original
         namedWindow("Original");
         imshow("Original", image);

         // Display  original Histograma
         processing::histDisplay(histogram, "Original Histogram");

         // Display imagen equilizada
         namedWindow("Imagen Ecualizada");
         imshow("Imagen Ecualizada",new_image);

         // Display histograma ecualizado
         processing::histDisplay(final, "Histograma Ecualizado");

 }








