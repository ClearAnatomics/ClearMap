
cdef enum:
  MAX_DIMS = 10

cdef struct Iterator:
  index_t n_dim
  index_t shape[MAX_DIMS]
  index_t coordinates[MAX_DIMS]
  index_t strides[MAX_DIMS]
  index_t backstrides[MAX_DIMS]

cdef void init_iterator(Iterator* iterator, index_t n_dim, index_t[:] shape, index_t[:] strides) nogil:
  cdef index_t d
  iterator[0].n_dim = n_dim;
  for d in range(n_dim):
    iterator[0].shape[d]       = shape[d];
    iterator[0].coordinates[d] = 0;
    iterator[0].strides[d]     = strides[d];
    iterator[0].backstrides[d] = strides[d] * shape[d];
    
    
cdef void init_subspace_iterator(Iterator* iterator, index_t axis, index_t iter_axis) nogil:
  cdef index_t d, d_last = 0;
  for d in range(iterator[0].n_dim):
    if d != axis and d != iter_axis:
      if d != d_last:
        iterator[0].shape[d_last]       = iterator[0].shape[d]
        iterator[0].strides[d_last]     = iterator[0].strides[d];
        iterator[0].backstrides[d_last] = iterator[0].backstrides[d];
      d_last += 1;
  iterator.n_dim = d_last;


cdef inline void line_correlate(source_t* source_line, sink_t* sink_line, double* kernel_pointer,
                                index_t k1, index_t k2, index_t line_shape,
                                index_t source_stride_axis, index_t sink_stride_axis,
                                index_t n_sub_iterations,
                                Iterator source_line_iterator, Iterator sink_line_iterator) nogil:
  
  cdef index_t i,j,m,d
  cdef double temp
  cdef source_t* source_point;
  cdef sink_t* sink_point
  
  for i in range(n_sub_iterations):
    
    source_point = source_line;
    sink_point   = sink_line;
    
    #calculate 1d correlation
    #left border
    for j in range(k1):
      temp = 0;
      for m in range(-j,k2):
        temp = temp + <double>source_point[m * source_stride_axis] * kernel_pointer[m];
      sink_point[0] = <sink_t>temp;
      sink_point += sink_stride_axis;
      source_point += source_stride_axis;
    #center
    for j in range(k1,line_shape-k2):
      temp = 0;
      for m in range(-k1,k2):
        temp = temp + <double>source_point[m * source_stride_axis] * kernel_pointer[m];
      sink_point[0] = <sink_t>temp;
      sink_point += sink_stride_axis;
      source_point += source_stride_axis;
    #right border
    for j in range(line_shape-k2,line_shape):
      temp = 0;
      for m in range(-k1,line_shape-j):
        temp = temp + <double>source_point[m * source_stride_axis] * kernel_pointer[m];
      sink_point[0] = <sink_t>temp;
      sink_point += sink_stride_axis;
      source_point += source_stride_axis;
      
    #set line pointers to next line
    for d in range(source_line_iterator.n_dim):
      if source_line_iterator.coordinates[d] < source_line_iterator.shape[d]:
        source_line_iterator.coordinates[d]+=1;
        source_line += source_line_iterator.strides[d];
        sink_line_iterator.coordinates[d]+=1;
        sink_line += sink_line_iterator.strides[d];
        break;
      else:
        source_line_iterator.coordinates[d] = 0;
        source_line -= source_line_iterator.backstrides[d];
        sink_line_iterator.coordinates[d] = 0;
        sink_line -= sink_line_iterator.backstrides[d];#correlate 1d

#cdef print_iterator(Iterator iterator):
#  print('dim=%d' % iterator.n_dim);
#  for d in range(iterator.n_dim):
#    print('d=%d, shape=%d, coords=%d, strides=%d, backstrides=%d' % (d, iterator.shape[d], iterator.coordinates[d], iterator.strides[d], iterator.backstrides[d]));
        

cpdef void correlate_1d(source_t[:] source, index_t[:] source_shape, index_t[:] source_strides, int axis,
                        sink_t[:] sink, index_t[:] sink_shape, index_t[:] sink_strides, double[:] kernel, int processes):
  
  cdef index_t n_dim = source_shape.shape[0];
  cdef index_t line_shape = source_shape[axis];
  
  cdef index_t source_stride_axis = source_strides[axis];
  cdef index_t sink_stride_axis   = sink_strides[axis];
  
  cdef index_t k = kernel.shape[0];
  cdef index_t k1 = k // 2;
  cdef index_t k2 = k - k1;
  assert(line_shape > k1 + k2)
  
  cdef source_t* source_pointer = &source[0];
  cdef sink_t*   sink_pointer   = &sink[0];
  cdef double*   kernel_pointer = &kernel[k1];
    
  #number of lines to iterate over
  cdef index_t iteration_axis;
  cdef index_t n_iterations = 0;
  cdef index_t n_subiterations = 1;
  cdef max_shape = 0;
  for d in range(n_dim):
    if d != axis:
      if max_shape < source_shape[d]:
        iteration_axis = d;
        n_iterations = source_shape[d];
  for d in range(n_dim):
    if d != axis and d != iteration_axis:
      n_subiterations *= source_shape[d];
  
  #initilaize iterators
  cdef Iterator source_iterator;
  init_iterator(&source_iterator, n_dim, source_shape, source_strides);
  
  cdef Iterator sink_iterator;
  init_iterator(&sink_iterator, n_dim, sink_shape, sink_strides);
  
  #initialize sub-space iterators excluding axis and the iteration_axis
  cdef Iterator source_line_iterator
  init_iterator(&source_line_iterator, n_dim, source_shape, source_strides);
  init_subspace_iterator(&source_line_iterator, axis, iteration_axis);
  
  cdef Iterator sink_line_iterator
  init_iterator(&sink_line_iterator, n_dim, sink_shape, sink_strides);
  init_subspace_iterator(&sink_line_iterator, axis, iteration_axis);
  
  #debug
  #print('k1,k2,ls=%d,%d,%d' % (k1,k2,line_shape));
  #print('source_iterator:');
  #print_iterator(source_iterator);
  #print('source_line');
  #print_iterator(source_line_iterator);
  #print('niter=%d, nsub=%d' % (n_iterations, n_subiterations))
  #print('axis=%d, iaxis=%d' % (axis, iteration_axis));
  
  cdef index_t iteration
  cdef source_t* source_line
  cdef sink_t*   sink_line
  
  with nogil, parallel(num_threads = processes):                 
    for iteration in prange(n_iterations, schedule = 'static'):
      #init pointers to first line
      source_line = source_pointer + source_iterator.strides[iteration_axis] * iteration;
      sink_line   = sink_pointer   +   sink_iterator.strides[iteration_axis] * iteration;
      
      line_correlate(source_line, sink_line, kernel_pointer, k1, k2, line_shape,
                     source_stride_axis, sink_stride_axis, n_subiterations,
                     source_line_iterator, sink_line_iterator);
  
  return;
        
        
        