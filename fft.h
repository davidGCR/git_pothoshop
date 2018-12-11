#ifndef FFT_H
#define FFT_H

#include "complex.h"
class FFT {

    bool test_pwr2(unsigned int n);
    static int rev_bits(unsigned int index,int size);

    public:
    static M_Complex* fft1(M_Complex *, unsigned int , int, bool inverse = false);
    static M_Complex** fft2(M_Complex**, int width, int height, bool inverse = false);
};

#endif // FFT_H
