#include "mainwindow.h"
#include "ui_mainwindow.h"
#include"wavelet.h"



MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{

    string image_name = "boy.bmp";
    image_path = PATH_IMAGES+image_name;
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}
void MainWindow::reloadImage(){
    raw_image = imread(image_path);
}
void MainWindow::keyPressEvent(QKeyEvent *event) // definition
{
    switch(event->key())
    {
        case Qt::Key_Escape:
            close();
            break;
        case Qt::Key_C:
            reloadImage();
            break;
        default:
            QMainWindow::keyPressEvent(event);
    }

//    if(event->key() == Qt::Key_Escape){

//    }

}

void MainWindow::on_btn_chooseImage_clicked()
{
    QString filename = QFileDialog::getOpenFileName(this,tr("Choose"),
                                                    "",tr("Images(*.png *.jpg *.jpeg *.bmp *.gif)"));
    image_path = filename.toStdString();
    cout<<"se eligio imagen: "<<image_path<<endl;

    raw_image = imread(image_path);
//    namedWindow("Imagen raw inicio");
//    imshow("Ima opencv",raw_image);

    if(QString::compare(filename,QString())!=0){
        QImage image;
        bool valid = image.load(filename);
        if(valid){
            image = image.scaledToWidth(ui->lbl_image->width(),Qt::SmoothTransformation);
            ui->lbl_image->setPixmap(QPixmap::fromImage(image));
        }
    }
}
void MainWindow::printProperties(Mat img){
    cout<<"size: "<<img.size()<<"\n channels: "<<img.channels()
       <<"\n type: "<<img.type()<<endl;

    if (img.depth() == CV_8U)
          cout << "CV_8U" << endl;
}

void MainWindow::on_load_img_opencv_clicked()
{
//    const String path = "/Users/davidchoqueluqueroman/Desktop/CURSOS-MASTER/IMAGENES/photoshop/images/"+image_name;
    Mat inputImage = imread(image_path);
    printProperties(inputImage);
    cv::cvtColor(inputImage,inputImage,CV_BGR2RGB); //Qt reads in RGB whereas CV in BGR
//    raw_image = inputImage;

    if (!inputImage.empty()) {
        QImage imdisplay((uchar*)inputImage.data, inputImage.cols, inputImage.rows,
                                 inputImage.step, QImage::Format_RGB888);
        imdisplay = imdisplay.scaledToWidth(ui->lbl_image->width(),Qt::SmoothTransformation);
        ui->lbl_image->setPixmap(QPixmap::fromImage(imdisplay));
     }
     else{
        cout<<"no se puede abrir imagen"<<endl;
     }
}
void print_histogram(Mat mat)
{
    cout<<"size: "<<mat.size()<<endl;
    for(int i=0; i<mat.size().height; i++)
    {
        cout << "i: "<<i<<"->[";
        for(int j=0; j<mat.size().width; j++)
        {
            cout << mat.at<double>(i,j);
            if(j != mat.size().width-1)
                cout << ", ";
            else
                cout << "]" << endl;
        }
    }
}

Mat MainWindow::calculateHistogram(Mat* img){
    // Separate the image in 3 places ( B, G and R )
      vector<Mat> bgr_planes;
      split(*(img), bgr_planes );
      // Establish the number of bins
        int histSize = 256;

        // Set the ranges ( for B,G,R) )
        float range[] = { 0, 256 } ;
        const float* histRange = { range };

        bool uniform = true; bool accumulate = false;

        Mat b_hist, g_hist, r_hist;

        // Compute the histograms:
        calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
        calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
        calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );

        // Draw the histograms for B, G and R
        int hist_w = 512; int hist_h = 400;
        int bin_w = cvRound( (double) hist_w/histSize );



        Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 255,255,255) );
        /// Normalize the result to [ 0, histImage.rows ]
          normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
          normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
          normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

          print_histogram(r_hist);
//          cout<<"hist 1: "<<r_hist.size()<<endl;
          /// Draw for each channel
          for( int i = 1; i < histSize; i++ )
          {
              line( histImage, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
                               Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
                               Scalar( 255, 0, 0), 2, 8, 0  );
              line( histImage, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
                               Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
                               Scalar( 0, 255, 0), 2, 8, 0  );
              line( histImage, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
                               Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
                               Scalar( 0, 0, 255), 2, 8, 0  );
          }

          double res = r_hist.at<double>(244,0)+r_hist.at<double>(252,0);
          cout<<"size: "<<r_hist.size()<<", rrrr: "<<res<<endl;
          return histImage;

}

void MainWindow::on_btn_show_histogram_clicked()
{
    Mat histogram = calculateHistogram(&raw_image);

    QImage imdisplay((uchar*)histogram.data, histogram.cols, histogram.rows,
                             histogram.step, QImage::Format_RGB888);
    imdisplay = imdisplay.scaledToWidth(ui->lbl_histogram->width(),Qt::SmoothTransformation);
    ui->lbl_histogram->setPixmap(QPixmap::fromImage(imdisplay));

}

void MainWindow::on_btn_eq_hist_clicked()
{
    processing::equalizarHistograma(image_path);
}

vector<Point>v_polygon;
vector<Point>quadrilat;
void onmouse(int event, int x, int y, int flags, void* param)
{
    if(event==CV_EVENT_LBUTTONDOWN)
    {
        int n_pts = v_polygon.size();
        if(n_pts<=2){
            Mat &img = *((Mat*)(param)); // 1st cast it back, then deref
            Point2f p = Point2f(x,y);
            circle(img,p,4,Scalar(0,250,0),1);
            v_polygon.push_back(p);
            cout<<"clickeoa aqui: "<<x<<", "<<y<<" poligono: "<<v_polygon.size()<<endl;
        }
    }
}
void onmouse_quad(int event, int x, int y, int flags, void* param)
{
    if(event==CV_EVENT_LBUTTONDOWN)
    {
        int n_pts = quadrilat.size();
        if(n_pts<=4){
            Mat &img = *((Mat*)(param)); // 1st cast it back, then deref
            Point2f p = Point2f(x,y);
            circle(img,p,4,Scalar(0,0,0),1);
            quadrilat.push_back(p);
            cout<<"clickeoa aqui2222: "<<x<<", "<<y<<" quad: "<<quadrilat.size()<<endl;
        }
    }
}
void MainWindow::on_btn_dibujar_puntos_clicked()
{
    // Display dashboard
    int k, size;
    int x1,y1,ancho,alto;
    Mat input_image, img_output,croppedImage;
//    cvtColor(raw_image, raw_image, CV_BGR2GRAY);
//    raw_image = raw_image;


    Rect Rec;
    namedWindow("Dibuje Puntos");
    setMouseCallback("Dibuje Puntos", onmouse, &raw_image);
    reloadImage();
    while(1){
        imshow("Dibuje Puntos",raw_image);
        size = v_polygon.size();
        if(size==2){
            x1 = (v_polygon[0].x < v_polygon[1].x)?v_polygon[0].x:v_polygon[1].x;
            y1 = (v_polygon[0].y < v_polygon[1].y)?v_polygon[0].y:v_polygon[1].y;

            ancho = abs(v_polygon[0].x-v_polygon[1].x);
            alto = abs(v_polygon[0].y-v_polygon[1].y);

            Rec =  Rect(x1, y1,ancho , alto);

            rectangle( raw_image, Rec, Scalar( 0, 55, 255 ), +1, 4 );
            cout<<"dibujar rectangulo... ";
            Mat ROI(raw_image, Rec);

//            // Copy the data into new matrix
            ROI.copyTo(croppedImage);

        }
        k = waitKey(10);

        if (k== 27 || k==13)
            break;
    }
    size = v_polygon.size();

    if(size == 2 && k==13){
        cout<<"size of image: "<<raw_image.size<<endl;
        Mat img_board(raw_image.rows, raw_image.cols, CV_8UC3, Scalar(255, 255, 255));
        cout<<"size of image cpy: "<<img_board.size<<endl;
        namedWindow("Dibuje Cuadrilatero");
        setMouseCallback("Dibuje Cuadrilatero", onmouse_quad, &img_board);
        while(1){
           imshow("Dibuje Cuadrilatero",img_board);
           k = waitKey(10);

           if (k== 27 || k==13)
               break;
        }
        if(k==13 && quadrilat.size()==4){

            for(int j=0;j<raw_image.rows;j++){
                for (int i=0;i<raw_image.cols;i++){
                    Vec3b pixel = raw_image.at<Vec3b>(i,j);
                            int b, g, r;
                            b = pixel[0];
                            g = pixel[1];
                            r = pixel[2];
                    if(i==100&&j==100){
                         cout<<"leer pixeeeeel main windows...("<<b<<","<<g<<","<<r<<")"<<endl;
                         break;
                    }

                 }
             }

            float u=0, v=0;
//            static void bilinearTransform(Mat image,Mat* respt,Rect, vector <Point>,float*,float*);
            Mat inv_img(raw_image.rows, raw_image.cols, CV_8UC3, Scalar(255, 255, 255));
            processing::bilinearTransform(raw_image,&img_board,&inv_img,Rec,quadrilat,&u, &v);

             cout << "transform222222: " << u<<","<<v<<endl;
            Point2f p = Point2f(u,v);
            circle(img_board,p,10,Scalar(0,0,0),1);
            imshow("Dibuje Cuadrilatero",img_board);

            dialogWindow = new DialogWindow();
            dialogWindow->setImgSrc(croppedImage);
            dialogWindow->setImgRpta1(img_board);
            dialogWindow->setImgRpta2(inv_img);
            dialogWindow->show();

    //        printProperties(raw_image);
    //        printProperties(croppedImage);
        }

    }
    v_polygon.clear();
    quadrilat.clear();
    destroyWindow("Dibuje Cuadrilatero");
    destroyWindow("Dibuje Puntos");
}

void MainWindow::on_btn_wavelet_clicked()
{
    image my;
    my.getim();
}

int reflect(int M, int x)
{
    if(x < 0)
    {
        return -x - 1;
    }
    if(x >= M)
    {
        return 2*M - x - 1;
    }
   return x;
}

int circular(int M, int x)
{
    if (x<0)
        return x+M;
    if(x >= M)
        return x-M;
   return x;
}


void noBorderProcessing(Mat src, Mat dst, float Kernel[][3])
{

    float sum;
    for(int y = 1; y < src.rows - 1; y++){
        for(int x = 1; x < src.cols - 1; x++){
            sum = 0.0;
            for(int k = -1; k <= 1;k++){
                for(int j = -1; j <=1; j++){
                    sum = sum + Kernel[j+1][k+1]*src.at<uchar>(y - j, x - k);
                }
            }
            dst.at<uchar>(y,x) = sum;
        }
    }
}

void refletedIndexing(Mat src, Mat dst, float Kernel[][3])
{
    float sum, x1, y1;
    for(int y = 0; y < src.rows; y++){
        for(int x = 0; x < src.cols; x++){
            sum = 0.0;
            for(int k = -1;k <= 1; k++){
                for(int j = -1;j <= 1; j++ ){
                    x1 = reflect(src.cols, x - j);
                    y1 = reflect(src.rows, y - k);
                    sum = sum + Kernel[j+1][k+1]*src.at<uchar>(y1,x1);
                }
            }
            dst.at<uchar>(y,x) = sum;
        }
    }
}

void circularIndexing(Mat src, Mat dst, float Kernel[][5])
{
    float sum, x1, y1;
    for(int y = 0; y < src.rows; y++){
        for(int x = 0; x < src.cols; x++){
            sum = 0.0;
            for(int k = -1;k <= 1; k++){
                for(int j = -1;j <= 1; j++ ){
                    x1 = circular(src.cols, x - j);
                    y1 = circular(src.rows, y - k);
                    sum = sum + Kernel[j+1][k+1]*src.at<uchar>(y1,x1);
                }
            }
            dst.at<uchar>(y,x) = sum;
        }
    }
}
void applyFilter(Mat &image, Mat &dst,float filter[][5]){


    int height = image.rows;
    int width = image.cols;
    int filterHeight = 5;
    int filterWidth = 5;
    int newImageHeight = height-filterHeight+1;
    int newImageWidth = width-filterWidth+1;
    int d,i,j,h,w;


    for (d=0 ; d<3 ; d++) {
        for (i=0 ; i<newImageHeight ; i++) {
            for (j=0 ; j<newImageWidth ; j++) {
                for (h=i ; h<i+filterHeight ; h++) {
                    for (w=j ; w<j+filterWidth ; w++) {
                        dst.at<uchar>(j,i) += filter[h-i][w-j]*dst.at<uchar>(w,h);
                    }
                }
            }
        }
    }
}

void createFilter(float gKernel[][5])
{
    // set standard deviation to 1.0
    float sigma = 1;
    float r, s = 2.0 * sigma * sigma;

    // sum is for normalization
    float sum = 0.0;

    // generate 5x5 kernel
    for (int x = -2; x <= 2; x++)
    {
        for(int y = -2; y <= 2; y++)
        {
            r = sqrt(x*x + y*y);
            gKernel[x + 2][y + 2] = (exp(-(r*r)/s))/(M_PI * s);
            sum += gKernel[x + 2][y + 2];
        }
    }

    // normalize the Kernel
    for(int i = 0; i < 5; ++i)
        for(int j = 0; j < 5; ++j)
            gKernel[i][j] /= sum;

}

// Computes the x component of the gradient vector
// at a given point in a image.
// returns gradient in the x direction
int xGradient(Mat image, int x, int y)
{
    return image.at<uchar>(y-1, x-1) +
                2*image.at<uchar>(y, x-1) +
                 image.at<uchar>(y+1, x-1) -
                  image.at<uchar>(y-1, x+1) -
                   2*image.at<uchar>(y, x+1) -
                    image.at<uchar>(y+1, x+1);
}

// Computes the y component of the gradient vector
// at a given point in a image
// returns gradient in the y direction

int yGradient(Mat image, int x, int y)
{
    return image.at<uchar>(y-1, x-1) +
                2*image.at<uchar>(y-1, x) +
                 image.at<uchar>(y-1, x+1) -
                  image.at<uchar>(y+1, x-1) -
                   2*image.at<uchar>(y+1, x) -
                    image.at<uchar>(y+1, x+1);
}


void MainWindow::on_btn_convolucion_clicked()
{

    reloadImage();
    Mat src = raw_image.clone();
//    cvtColor(src, src, CV_BGR2GRAY);

    Mat dst;
    Mat kernel;

    dst = src.clone();
    for(int y = 0; y < src.rows; y++)
        for(int x = 0; x < src.cols; x++)
            dst.at<uchar>(y,x) = 0;

    processing::convolute(src,dst,kernel);
    namedWindow("final");
    imshow("final", dst);

//    namedWindow("initial");
//    imshow("initial", src);

//        float Kernel[5][5];
//        createFilter(Kernel);

//        for(int i = 0; i < 5; ++i) // loop to display the generated 5 x 5 Gaussian filter
//            {
//                for (int j = 0; j < 5; ++j)
//                    cout<<Kernel[i][j]<<"\t";
//                cout<<endl;
//            }
//          float Kernel[3][3] = {
//                                {1/9.0, 1/9.0, 1/9.0},
//                                {1/9.0, 1/9.0, 1/9.0},
//                                {1/9.0, 1/9.0, 1/9.0}
//                               };

//        circularIndexing(src, dst, Kernel);

//        applyFilter(src, dst, Kernel);

//        QImage imdisplay((uchar*)dst.data, dst.cols, dst.rows,
//                                 dst.step, QImage::Format_RGB888);
//        ui->lbl_image_dst->setPixmap(QPixmap::fromImage(imdisplay));
//        ui->lbl_histogram->setScaledContents( true );

            namedWindow("final");
            imshow("final", dst);

//            namedWindow("initial");
//            imshow("initial", src);

}



void MainWindow::on_btn_bordes_clicked()
{
    reloadImage();
    Mat src = raw_image.clone();
    cvtColor(src, src, CV_BGR2GRAY);

    Mat dst;
    Mat kernel;

    dst = src.clone();
    for(int y = 0; y < src.rows; y++)
        for(int x = 0; x < src.cols; x++)
            dst.at<uchar>(y,x) = 0;
    int gx, gy, sum;

    for(int y = 1; y < src.rows - 1; y++){
                for(int x = 1; x < src.cols - 1; x++){
                    gx = xGradient(src, x, y);
                    gy = yGradient(src, x, y);
                    sum = abs(gx) + abs(gy);
                    sum = sum > 255 ? 255:sum;
                    sum = sum < 0 ? 0 : sum;
                    dst.at<uchar>(y,x) = sum;
                }
    }
    namedWindow("final");
    imshow("final", dst);
}

void GammaCorrection(Mat& src, Mat& dst, float fGamma)

{

    unsigned char lut[256];

    for (int i = 0; i < 256; i++)

    {

        lut[i] = saturate_cast<uchar>(pow((float)(i / 255.0), fGamma) * 255.0f);

    }

    dst = src.clone();

    const int channels = dst.channels();

    switch (channels)

    {

    case 1:

    {

        MatIterator_<uchar> it, end;

        for (it = dst.begin<uchar>(), end = dst.end<uchar>(); it != end; it++)

            *it = lut[(*it)];

        break;

    }

    case 3:

    {

        MatIterator_<Vec3b> it, end;

        for (it = dst.begin<Vec3b>(), end = dst.end<Vec3b>(); it != end; it++)

        {

            (*it)[0] = lut[((*it)[0])];

            (*it)[1] = lut[((*it)[1])];

            (*it)[2] = lut[((*it)[2])];

        }

        break;

    }

    }

}

void MainWindow::on_btn_gamma_clicked()
{
    reloadImage();
    Mat src = raw_image.clone();
    Mat dst;
    GammaCorrection(src,dst,2);

    namedWindow("final");
    imshow("final", dst);
}
