
#include "LineIteration.h"

int main() {
  int n_dim = 3;
  int shape[3] = {10,20,30};
  int strides[3] = {0, 10, 200};
  int size = 10*20*30;
  
  double* source = (double*)malloc(size*sizeof(double));
  double* sink   = (double*)malloc(size*sizeof(double));
  
  double* kernel = (double*)malloc(4*sizeof(double));
  int k1 = 2, k2 = 2;
  
  int axis = 0;
  int n_lines = 20*30;
  int index = 0;
  
  int max_buffer_size = 256000;
  int max_buffer_lines = n_lines;

  line_correlate(source, sink, n_dim, 
               shape, strides, shape, strides,
               kernel, k1, k2, 
               axis, index, n_lines,
               max_buffer_size, max_buffer_lines);

}