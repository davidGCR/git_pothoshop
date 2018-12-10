#include "dialogwindow.h"
#include "ui_dialogwindow.h"

DialogWindow::DialogWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::DialogWindow)
{
    ui->setupUi(this);
}

DialogWindow::~DialogWindow()
{
    delete ui;
}
void DialogWindow::setImgSrc(Mat img_rpta){
    img_src = img_rpta;
    cv::cvtColor(img_src,img_src,CV_BGR2RGB);
    QImage imdisplay((uchar*)img_src.data, img_src.cols, img_src.rows,
                             img_src.step, QImage::Format_RGB888);
    ui->lbl_img_raw->setPixmap(QPixmap::fromImage(imdisplay));
    ui->lbl_img_raw->setScaledContents( true );

}
void DialogWindow::setImgRpta1(Mat img_rpta){
    img_rpta1 = img_rpta;
    cv::cvtColor(img_rpta1,img_rpta1,CV_BGR2RGB);
    QImage imdisplay((uchar*)img_rpta1.data, img_rpta1.cols, img_rpta1.rows,
                             img_rpta1.step, QImage::Format_RGB888);
    ui->lbl_img_rpta1->setPixmap(QPixmap::fromImage(imdisplay));
    ui->lbl_img_rpta1->setScaledContents( true );

}
void DialogWindow::setImgRpta2(Mat img_rpta){
    img_rpta2 = img_rpta;
    cv::cvtColor(img_rpta2,img_rpta2,CV_BGR2RGB);
    QImage imdisplay((uchar*)img_rpta2.data, img_rpta2.cols, img_rpta2.rows,
                             img_rpta2.step, QImage::Format_RGB888);
    ui->lbl_img_rpta2->setPixmap(QPixmap::fromImage(imdisplay));
    ui->lbl_img_rpta2->setScaledContents( true );
}
void DialogWindow::on_btn_operation_clicked()
{

}
void DialogWindow::keyPressEvent(QKeyEvent *event) // definition
{
    switch(event->key())
    {
        case Qt::Key_Escape:
            close();
            break;
//        case Qt::Key_C:
//            reloadImage();
//            break;
        default:
            QMainWindow::keyPressEvent(event);
    }

//    if(event->key() == Qt::Key_Escape){

//    }

}
