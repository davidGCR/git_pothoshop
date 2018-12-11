#ifndef COMPLEX_H
#define COMPLEX_H


#include <iostream>
#include <cmath>

using namespace std;

class M_Complex
{
public:
    double real;
    double imag;
public:
    M_Complex();
    M_Complex(const M_Complex &v);
    M_Complex(const double &real, const double &imag);
    void operator = (M_Complex v);
    void operator = (double v);
    M_Complex operator - (M_Complex v);
    M_Complex operator + (M_Complex v);
    M_Complex operator * (M_Complex v);
    M_Complex operator * (double v);
    void operator *= (M_Complex v);
    void operator *= (double v);
    M_Complex* operator += (M_Complex v);
    M_Complex* operator -= (M_Complex v);
    void operator /= (M_Complex v);
    double mag();
    double squaredSum();
    double angle();

};

ostream& operator<<(ostream& output, const M_Complex& v);

#endif // COMPLEX_H
