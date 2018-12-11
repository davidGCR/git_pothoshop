#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QFileDialog>

#include <opencv2/opencv.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "constants.h"
#include "processing.h"
 #include <QKeyEvent>
#include<vector>
#include "dialogwindow.h"
#include <cmath>
#include<iostream>
#include "complex.h"
#include "fft.h"
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
    void printProperties(Mat );
    Mat calculateHistogram(Mat*);
    void reloadImage();
    //static void onmouse(int event, int x, int y, int flags, void* param);
//    static vector<Point> v_polygon;


private slots:
    void on_btn_chooseImage_clicked();

    void on_load_img_opencv_clicked();

    void on_btn_show_histogram_clicked();

    void on_btn_eq_hist_clicked();

    void on_btn_dibujar_puntos_clicked();

    void on_btn_wavelet_clicked();

    void on_btn_convolucion_clicked();

    void on_btn_bordes_clicked();

    void on_btn_gamma_clicked();

    void on_btn_video_clicked();

    void on_btn_fft_clicked();

private:
    Ui::MainWindow *ui;
    Mat raw_image;
    string image_path;
    DialogWindow* dialogWindow;

    void keyPressEvent(QKeyEvent *event);
};

#endif // MAINWINDOW_H
