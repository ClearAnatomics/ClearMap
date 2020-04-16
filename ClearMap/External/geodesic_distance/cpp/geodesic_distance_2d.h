#include <iostream>
#include <vector>
using namespace std;

struct Point2D
{
    float distance;
    int w;
    int h;
};
void geodesic2d_fast_marching(const float * img, const unsigned char * seeds, float * distance, int height, int width);

void geodesic2d_raster_scan(const float * img, const unsigned char * seeds, float * distance, int height, int width);