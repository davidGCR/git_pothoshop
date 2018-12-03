#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QFileDialog>

#include <opencv2/opencv.hpp>
#include<iostream>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
using namespace cv;
using namespace std;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    void printProperties(Mat* );
    Mat histogramEcualization(Mat*,int);
    Mat calculateHistogram(Mat*);

private slots:
    void on_btn_chooseImage_clicked();

    void on_load_img_opencv_clicked();

    void on_btn_show_histogram_clicked();

private:
    Ui::MainWindow *ui;
    Mat raw_image;
    string image_name;
};

#endif // MAINWINDOW_H
