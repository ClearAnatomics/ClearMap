#include <iostream>
#include <vector>
using namespace std;

// for 2d images
template<typename T>
T get_pixel(const T * data, int height, int width, int h, int w);

template<typename T>
void set_pixel(T * data,  int height, int width, int h, int w, T value);

// for 3d images
template<typename T>
T get_pixel(const T * data, int depth, int height, int width, int d, int h, int w);

template<typename T>
void set_pixel(T * data,  int depth, int height, int width, int d, int h, int w, T value);