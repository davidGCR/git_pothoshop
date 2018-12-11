#include "complex.h"

M_Complex::M_Complex()
{
    real = 0;
    imag = 0;
}

M_Complex::M_Complex(const M_Complex &v)
{
    real = v.real;
    imag = v.imag;
}

M_Complex::M_Complex(const double &real, const double &imag)
{
    this->real = real;
    this->imag = imag;
}

void M_Complex::operator=(M_Complex v)
{
    real = v.real;
    imag = v.imag;
}


void M_Complex::operator=(double v)
{
    real = v;
    imag = v;
}

M_Complex M_Complex::operator*(double v)
{
    return M_Complex(real*v, imag*v);
}

M_Complex M_Complex::operator+(M_Complex v)
{
    return M_Complex(real+v.real, imag+v.imag);
}

M_Complex M_Complex::operator-(M_Complex v)
{
    return M_Complex(real-v.real, imag-v.imag);
}

M_Complex M_Complex::operator*(M_Complex v)
{
    return M_Complex(real*v.real - imag*v.imag, real*v.imag + imag*v.real);
}

M_Complex* M_Complex::operator+=(M_Complex v)
{
    real += v.real;
    imag += v.imag;
    return this;
}

M_Complex* M_Complex::operator-=(M_Complex v)
{
    real -= v.real;
    imag -= v.imag;
    return this;
}

void M_Complex::operator*=(M_Complex v)
{
    real *= v.real;
    imag *= v.imag;
}

void M_Complex::operator*=(double v)
{
    real *= v;
    imag *= v;
}


double M_Complex::mag()
{
    return sqrt(real*real + imag*imag);
}

double M_Complex::squaredSum()
{
    return real*real + imag*imag;
}

double M_Complex::angle()
{
    double a = atan2(real, imag);
    if(a < 0) {
        a = (M_PI * 2) + a;
    }
    return a;
}

ostream& operator<<(ostream& output, const M_Complex& v) {
    output.setf(ios::fixed, ios::floatfield);
    output.precision(2);
    if(abs(v.imag) > 0.001){
        if(v.imag < 0)
            output  << v.real << " - " << abs(v.imag)<<"i";
        else
            output  << v.real << " + " << abs(v.imag)<<"i";
    }else{
        if (abs(v.real) < 0.001)
            output << abs(v.real);
            else
                output << v.real;
    }
    return output;
}
