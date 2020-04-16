#include "util.h"
#include <iostream>
// for 2d images
template<typename T>
T get_pixel(const T * data, int height, int width, int h, int w)
{
    return data[h * width + w];
}

template<typename T>
void set_pixel(T * data,  int height, int width, int h, int w, T value)
{
    data[h * width + w] = value;
}

template
float get_pixel<float>(const float * data, int height, int width, int h, int w);

template
int get_pixel<int>(const int * data, int height, int width, int h, int w);

template
unsigned char get_pixel<unsigned char>(const unsigned char * data, int height, int width, int h, int w);


template
void set_pixel<float>(float * data, int height, int width, int h, int w, float value);

template
void set_pixel<int>(int * data, int height, int width, int h, int w, int value);

template
void set_pixel<unsigned char>(unsigned char * data, int height, int width, int h, int w, unsigned char value);

// for 3d images
template<typename T>
T get_pixel(const T * data, int depth, int height, int width, int d, int h, int w)
{
    return data[(d*height + h) * width + w];
}

template<typename T>
void set_pixel(T * data, int depth, int height, int width, int d, int h, int w, T value)
{
    data[(d*height + h) * width + w] = value;
}

template
float get_pixel<float>(const float * data, int depth, int height, int width, int d, int h, int w);

template
int get_pixel<int>(const int * data, int depth, int height, int width, int d, int h, int w);

template
unsigned char get_pixel<unsigned char>(const unsigned char * data,
                                       int depth, int height, int width,
                                       int d, int h, int w);


template
void set_pixel<float>(float * data, int depth, int height, int width, int d, int h, int w, float value);

template
void set_pixel<int>(int * data, int depth, int height, int width, int d, int h, int w, int value);

template
void set_pixel<unsigned char>(unsigned char * data,
                              int depth, int height, int width,
                              int d, int h, int w, unsigned char value);