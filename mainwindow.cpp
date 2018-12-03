#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "constants.h"


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
//        Mat inputImage = imread("/Users/davidchoqueluqueroman/Desktop/CURSOS-MASTER/IMAGENES/photoshop/images/img.jpg");
////        cvtColor(inputImage, inputImage, CV_BGR2RGB);

//        if (!inputImage.empty()) {
//            imshow("window", inputImage);
//        }
//        else{
//            cout<<"no se puede abrir imagen"<<endl;
//        }
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_btn_chooseImage_clicked()
{
    QString filename = QFileDialog::getOpenFileName(this,tr("Choose"),
                                                    "",tr("Images(*.png *.jpg *.jpeg *.bmp *.gif)"));
    if(QString::compare(filename,QString())!=0){
        QImage image;
        bool valid = image.load(filename);
        if(valid){
            image = image.scaledToWidth(ui->lbl_image->width(),Qt::SmoothTransformation);
            ui->lbl_image->setPixmap(QPixmap::fromImage(image));
        }
    }
}
void MainWindow::printProperties(Mat* img){
    cout<<"size: "<<img->size()<<"\n channels: "<<img->channels()
       <<"\n type: "<<img->type()<<endl;

    if (img->depth() == CV_8U)
          cout << "CV_8U" << endl;
}

void MainWindow::on_load_img_opencv_clicked()
{
    image_name = "boy.bmp";
//    const String path = "/Users/davidchoqueluqueroman/Desktop/CURSOS-MASTER/IMAGENES/photoshop/images/"+image_name;
    Mat inputImage = imread(PATH_IMAGES+image_name);
    printProperties(&inputImage);
    cv::cvtColor(inputImage,inputImage,CV_BGR2RGB); //Qt reads in RGB whereas CV in BGR
    raw_image = inputImage;

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
Mat MainWindow::histogramEcualization(Mat* p_histogram, int total_pixels){
    int f[256];
    f[0]=0;
//    int acumulado = (*p_histogram)[0];
//    for(int i=1;i<254;i++){

//    }
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
