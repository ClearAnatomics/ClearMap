// Iterate over lines in a array using a line buffer
//__author__    = 'Christoph Kirst <ckirst@rockefeller.edu>'
//__license__   = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
//__copyright__ = 'Copyright (c) 2019 by Christoph Kirst'

#include <stdlib.h>
//#include <iostream>
//#include <stdio.h>

#include <numpy/arrayobject.h>

#define MAX_DIMS 10

typedef npy_intp index_t;

struct Iterator {
  index_t n_dim;
  index_t n_dim_m1;
  index_t shape[MAX_DIMS];
  index_t shape_m1[MAX_DIMS];
  index_t coordinates[MAX_DIMS];
  index_t strides[MAX_DIMS];
  index_t backstrides[MAX_DIMS];
};

//void iterator_print(Iterator* iterator) {
// std::cout << "Iterator [" << iterator->n_dim << "," << iterator->n_dim_m1 << "]\n";
// for (int d = 0; d < iterator->n_dim; d++) {
//        std::cout << "shape=" << iterator-> shape[d] << " ";
//        std::cout << "shape_m1=" << iterator-> shape_m1[d] << " "; 
//        std::cout << "coord=" << iterator-> coordinates[d] << " ";
//        std::cout << "strides=" << iterator-> strides[d] << " ";
//        std::cout << "backstr=" << iterator-> backstrides[d] << "\n";
//      }
//      std::cout << "\n";
//}

void init_iterator(Iterator* iterator, index_t n_dim, index_t* shape, index_t* strides) {
  iterator->n_dim = n_dim;
  iterator->n_dim_m1 = n_dim - 1;
  for(index_t d = 0; d < n_dim; d++) {
    iterator->shape[d]       = shape[d];
    iterator->shape_m1[d]    = shape[d] - 1;  
    iterator->coordinates[d] = 0;
    iterator->strides[d]     = strides[d];
    iterator->backstrides[d] = strides[d] * (shape[d] - 1);
  }
};
    
void init_subspace_iterator(Iterator* iterator, index_t axis) {
  index_t d_last = 0;
  //std::cout << "sub init " << axis << iterator->n_dim << std::endl;
  for(int d = 0; d < iterator->n_dim; ++d) {
    //std::cout << "d=" << d << std::endl;
    if (d != axis) {
      if (d != d_last) {
        iterator->shape[d_last]       = iterator->shape[d];
        iterator->shape_m1[d_last]    = iterator->shape_m1[d];
        iterator->strides[d_last]     = iterator->strides[d];
        iterator->backstrides[d_last] = iterator->backstrides[d];
      }
      d_last += 1;
    }
    //std::cout << d << ", " << d_last << std::endl;
  }
  iterator->n_dim = d_last;
  iterator->n_dim_m1 = d_last-1;
  
  //std::cout << "final: " << iterator->n_dim << std::endl << std::endl;
};
  
inline index_t iterator_iterations(Iterator* iterator) {
  index_t n = 1;
  for(index_t d = 0; d < iterator->n_dim; d++){
    n *= iterator->shape[d];
  }
  return n;
};  

void iterator_index_to_coordinates(Iterator* iterator, index_t index){
  index_t r = index;
  index_t n = iterator_iterations(iterator);
  for(index_t d = iterator->n_dim_m1; d >= 0; d--){
    n /= iterator->shape[d];
    iterator->coordinates[d] = r / n;
    r = r % n;
  }
  return;
};

index_t iterator_index_from_coordinates(Iterator* iterator){
  index_t index = 0;
  index_t n = 1;
  for(index_t d = 0; d < iterator->n_dim; d++){
    index += iterator->coordinates[d] * n;
    n *= iterator->shape[d];
  }
  return index;
};

index_t iterator_iterations_left(Iterator* iterator) {
  return iterator_iterations(iterator) - iterator_index_from_coordinates(iterator);
};
  
#define ITERATOR_NEXT(iterator, pointer)                                     \
{                                                                            \
  for(index_t d = iterator.n_dim_m1; d >= 0; d--) {                          \
    if (iterator.coordinates[d] < iterator.shape_m1[d]) {                    \
      iterator.coordinates[d]++;                                             \
      pointer += iterator.strides[d];                                        \
      break;                                                                 \
    } else {                                                                 \
      iterator.coordinates[d] = 0;                                           \
      pointer -= iterator.backstrides[d];                                    \
    }                                                                        \
  }                                                                          \
}
    
template<typename T>
inline T* iterator_coordinates_to_pointer(Iterator* iterator, T* source) { 
  T* pointer = source;                                                        
  for(index_t d = 0; d < iterator->n_dim; d++) {
    pointer += iterator->coordinates[d] * iterator->strides[d];                           
  }
  return pointer;              
}; 


//line buffers
template<typename T>
struct LineBuffer {
  double *buffer;
  index_t n_buffer_lines;     //number of lines in the buffer
  index_t max_buffer_lines;   //max number of lines the buffer can store
  index_t buffer_line;        //current line position in the buffer
  index_t buffer_stride;
  
  T* source;
  index_t n_source_lines;     //number of lines to process
  index_t source_line;        //current position in the source
  index_t source_stride_axis; 
  
  Iterator iterator;          //iterator over lines
};

//template<typename T>
//void buffer_print(LineBuffer<T>* buffer) {
//  std::cout << "Linebuffer:" << std::endl;
//  std::cout << "n_buffer_lines     = " << buffer->n_buffer_lines << std::endl;
//  std::cout << "max_buffer_lines   = " << buffer->max_buffer_lines << std::endl;
//  std::cout << "buffer_line        = " << buffer->buffer_line << std::endl;
//  std::cout << "buffer_stride      = " << buffer->buffer_stride << std::endl;
//  std::cout << "n_source_lines     = " << buffer->n_source_lines << std::endl;
//  std::cout << "source_line        = " << buffer->source_line << std::endl;
//  std::cout << "source_stride_axis = " << buffer->source_stride_axis << std::endl;
//  iterator_print(&(buffer->iterator));
//}

template<typename T>
void init_line_buffer(LineBuffer<T>* buffer, T* source, index_t n_dim, 
                      index_t* shape, index_t* strides, 
                      index_t axis, index_t index, index_t n_lines, 
                      index_t max_buffer_size, index_t max_buffer_lines) {
  //init iterator
  Iterator* iterator = &(buffer->iterator);
  init_iterator(iterator, n_dim, shape, strides);
  init_subspace_iterator(iterator, axis);  
  iterator_index_to_coordinates(iterator, index);
  
  //init source info
  buffer->source_stride_axis = strides[axis];
  buffer->n_source_lines = n_lines;
  if (n_lines < iterator_iterations_left(iterator)) {
    buffer->n_source_lines = iterator_iterations_left(iterator);
  }
  buffer->source = iterator_coordinates_to_pointer(iterator, source);
  buffer->source_line = 0;
  
  //init buffer   
  index_t max_lines = buffer->n_source_lines < max_buffer_lines ? buffer->n_source_lines : max_buffer_lines;
  index_t line_shape = shape[axis];
  if (max_lines * line_shape > max_buffer_size) {
    max_lines = max_buffer_size / line_shape;
    max_lines = max_lines > 0 ? max_lines : 1;
  }
  buffer->max_buffer_lines = max_lines;
  buffer->buffer_stride = line_shape;
  buffer->n_buffer_lines = 0;
  buffer->buffer_line = 0; 
};

template<typename T>
void buffer_from_source(LineBuffer<T>* buffer) {
  index_t line_shape = buffer->buffer_stride;
  index_t source_stride_axis = buffer->source_stride_axis;
   
  double* b = buffer->buffer;
  T*      s;
  index_t i;
  index_t n_lines = 0;
   
  while (buffer->source_line < buffer->n_source_lines && n_lines < buffer->max_buffer_lines) {  
    s = buffer->source;    
    for (i = 0; i < line_shape; ++i) {                                
      b[i] = (double)*s;
      s += source_stride_axis;               
    }   
    ITERATOR_NEXT(buffer->iterator, buffer->source);
    b += line_shape;   
    ++(buffer->source_line);
    ++(n_lines);
  }
  
  buffer->buffer_line = 0;
  buffer->n_buffer_lines = n_lines;
};

template<typename T>
void buffer_to_sink(LineBuffer<T>* buffer) {
  index_t line_shape = buffer->buffer_stride;
  index_t source_stride_axis = buffer->source_stride_axis;
   
   double* b = buffer->buffer;
   T*      s;
   index_t l,i;
   
   for(l = 0; l < buffer->max_buffer_lines; ++l) { 
     if (buffer->source_line == buffer->n_source_lines)
       break; 
     s = buffer->source;    
     for (i = 0; i < line_shape; ++i) {                                
        *s = (T)b[i];
        s += source_stride_axis;               
     }   
     ITERATOR_NEXT(buffer->iterator, buffer->source);
     b += line_shape;   
     ++(buffer->source_line);
  }
};



template<typename T, typename S>
void line_correlate(T* source, S* sink, index_t n_dim, 
                    index_t* source_shape, index_t* source_strides, 
                    index_t* sink_shape, index_t* sink_strides,
                    double* kernel, index_t kernel_shape_1, index_t kernel_shape_2, 
                    index_t axis, index_t index, index_t n_lines,
                    index_t max_buffer_size, index_t max_buffer_lines) {                     
  //setup line buffers  
  LineBuffer<T> source_line_buffer;
  LineBuffer<S> sink_line_buffer;
  init_line_buffer(&source_line_buffer, source, n_dim, source_shape, source_strides, 
                   axis, index, n_lines, max_buffer_size, max_buffer_lines);
  //iterator_print(&source_line_buffer.iterator);               
                   
  init_line_buffer(&sink_line_buffer, sink, n_dim, sink_shape, sink_strides,   
                   axis, index, n_lines, max_buffer_size, max_buffer_lines);
  //iterator_print(&source_line_buffer.iterator);   
  
  //allocate buffersckirst@ChristophsLaptop:~/Science/Projects/WholeBrainClearing/Vasculature/Analysis/ClearMap/ClearMap/ParallelProcessing/DataProcessing>
  index_t line_shape = source_shape[axis];
  double* source_buffer = (double*)malloc(source_line_buffer.max_buffer_lines * line_shape * sizeof(double));
  source_line_buffer.buffer = source_buffer;
  
  //TODO: check != NULL
  double* sink_buffer = (double*)malloc(source_line_buffer.max_buffer_lines * line_shape * sizeof(double));
  sink_line_buffer.buffer = sink_buffer;
  //TODO: check != NULL
  
  //iterate over lines  
  index_t i,j,m;
  index_t line_shape_m_kernel_shape_2 = line_shape - kernel_shape_2;
  double* source_line;
  double* sink_line;
  double temp;
  
  do {
    buffer_from_source(&source_line_buffer);
    //buffer_print(&source_line_buffer);
    //buffer_print(&sink_line_buffer);
    
    source_line = source_buffer;
    sink_line   = sink_buffer;
    
    for (i = 0; i < source_line_buffer.n_buffer_lines; ++i){       
      //left border
      for(j = 0; j < kernel_shape_1; ++j) {
        temp = 0;
        for(m = -j; m < kernel_shape_2; ++m) {
          temp += source_line[m] * kernel[m];
          
        }
        sink_line[j] = temp;
        ++source_line;
      }
      //center
      for(j = kernel_shape_1; j < line_shape_m_kernel_shape_2; ++j) {
        temp = 0;
        for(m = -kernel_shape_1; m < kernel_shape_2; ++m) {
          temp += source_line[m] * kernel[m];
        }
        sink_line[j] = temp;
        ++source_line;
      }
      //left border
      for(j = line_shape_m_kernel_shape_2; j < line_shape; ++j) {
        temp = 0;
        for(m = -kernel_shape_1; m < line_shape - j; ++m) {
          temp += source_line[m] * kernel[m];
        }
        sink_line[j] = temp;
        ++source_line;
      }
      
      //std::cout << "line i= " << i << " [";
      //for(j=0; j < line_shape; ++j)
      //  std::cout << source_line[j-line_shape] << ",";
      //std::cout << "]" << std::endl;
      //std::cout<< "sink line = [";
      //for(j=0; j < line_shape; ++j)
      //  std::cout << sink_line[j] << ",";
      //std::cout << "]" << std::endl;
      
      //next line
      sink_line   += line_shape;
    }
    
    buffer_to_sink(&sink_line_buffer);
    //buffer_print(&sink_line_buffer);
    
  } while(source_line_buffer.source_line < source_line_buffer.n_source_lines);
  
  free((char*)source_buffer);
  free((char*)sink_buffer);
  return;
};
