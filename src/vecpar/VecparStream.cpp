#include "VecparStream.hpp"

template <class T>
VecparStream<T>::VecparStream(const int ARRAY_SIZE, int device)
{
    array_size = ARRAY_SIZE;

    // Allocate on the host
    this->a = new vecmem::vector<T>(array_size, &memoryResource);
    this->b = new vecmem::vector<T>(array_size, &memoryResource);
    this->c = new vecmem::vector<T>(array_size, &memoryResource);
}

template <class T>
VecparStream<T>::~VecparStream()
{
    free(a);
    free(b);
    free(c);
}

template <class T>
void VecparStream<T>::init_arrays(T initA, T initB, T initC)
{
    int array_size = this->array_size;

    for (int i = 0; i < array_size; i++)
    {
        a->at(i) = initA;
        b->at(i) = initB;
        c->at(i) = initC;
    }

#if defined(VECPAR_GPU) && defined(DEFAULT)
    d_a = copy_tool.to(vecmem::get_data(*a),
                  dev_mem,
                  vecmem::copy::type::host_to_device);
    d_b = copy_tool.to(vecmem::get_data(*b),
                  dev_mem,
                  vecmem::copy::type::host_to_device);
    d_c = copy_tool.to(vecmem::get_data(*c),
                  dev_mem,
                  vecmem::copy::type::host_to_device);
#endif
}

template <class T>
void VecparStream<T>::read_arrays(std::vector<T>& h_a, std::vector<T>& h_b, std::vector<T>& h_c)
{

#if defined(VECPAR_GPU) && defined(DEFAULT)
    copy_tool(d_a, *a, vecmem::copy::type::device_to_host);
    copy_tool(d_b, *b, vecmem::copy::type::device_to_host);
    copy_tool(d_c, *c, vecmem::copy::type::device_to_host);
#endif

    for (int i = 0; i < array_size; i++)
    {
        h_a[i] = a->at(i);
        h_b[i] = b->at(i);
        h_c[i] = c->at(i);
    }
}

template <class T>
void VecparStream<T>::copy()
{
#if defined(SINGLE_SOURCE) // gpu+managed, cpu
    vecpar_copy<T> algorithm;
    vecpar::parallel_algorithm(algorithm, memoryResource, *c, *a);
#else
    #ifdef VECPAR_GPU
        vecpar::cuda::parallel_map(
                array_size,
                [=] __device__ (int idx,
                vecmem::data::vector_view<T> &c_view,
                vecmem::data::vector_view<T> &a_view) {
            vecmem::device_vector<T> dc(c_view);
            vecmem::device_vector<T> da(a_view);
            dc[idx] = da[idx] ;
        },
        vecmem::get_data(d_c),
        vecmem::get_data(d_a));
    #else // lambda
        vecpar::omp::parallel_map(array_size,
                              [&] (int idx)  { c->at(idx) = a->at(idx);});
    #endif
#endif
}

template <class T>
void VecparStream<T>::mul()
{
    const T scalar = startScalar;
#if defined(SINGLE_SOURCE)
    vecpar_mul<T> algorithm;
    vecpar::parallel_algorithm(algorithm, memoryResource, *b, *c, scalar);
#else
    #ifdef VECPAR_GPU
        vecpar::cuda::parallel_map(
                array_size,
                [=] __device__ (int idx,
                vecmem::data::vector_view<T> &b_view,
                vecmem::data::vector_view<T> &c_view,
                T dscalar) {
            vecmem::device_vector<T> db(b_view);
            vecmem::device_vector<T> dc(c_view);
            db[idx] = dscalar * dc[idx] ;
           },
        vecmem::get_data(d_b),
        vecmem::get_data(d_c),
        scalar);
    #else
        vecpar::omp::parallel_map(array_size,
                              [&] (int idx)  { b->at(idx) = scalar * c->at(idx);});
    #endif
#endif
}

template <class T>
void VecparStream<T>::add()
{
#if defined(SINGLE_SOURCE)
    vecpar_add<T> algorithm;
    vecpar::parallel_algorithm(algorithm, memoryResource, *c, *a, *b);
#else //defined (DEFAULT)
    #ifdef VECPAR_GPU
        vecpar::cuda::parallel_map(
                array_size,
                [=] __device__ (int idx,
                vecmem::data::vector_view<T> &a_view,
                vecmem::data::vector_view<T> &b_view,
                vecmem::data::vector_view<T> &c_view) {
            vecmem::device_vector<T> da(a_view);
            vecmem::device_vector<T> db(b_view);
            vecmem::device_vector<T> dc(c_view);
            dc[idx] = da[idx] + db[idx] ;
            },
        vecmem::get_data(d_a),
        vecmem::get_data(d_b),
        vecmem::get_data(d_c));
    #else
        vecpar::omp::parallel_map(array_size,
                              [&] (int idx)  { c->at(idx) = a->at(idx) + b->at(idx);});
    #endif
#endif
}

template <class T>
void VecparStream<T>::triad()
{
    const T scalar = startScalar;
    int array_size = this->array_size;

    #if defined(SINGLE_SOURCE)
        vecpar_triad<T> algorithm;
        vecpar::parallel_algorithm(algorithm, memoryResource, *a, *b, *c, scalar);
    #else //defined (DEFAULT)
        #if defined (VECPAR_GPU)
            vecpar::cuda::parallel_map(
                    array_size,
                    [=] __device__ (int idx,
                                    vecmem::data::vector_view<T> &a_view,
                                    vecmem::data::vector_view<T> &b_view,
                                    vecmem::data::vector_view<T> &c_view,
                                    T dscalar) {
                vecmem::device_vector<T> da(a_view);
                vecmem::device_vector<T> db(b_view);
                vecmem::device_vector<T> dc(c_view);
                da[idx] = db[idx] + dscalar * dc[idx];
            },
            vecmem::get_data(d_a),
            vecmem::get_data(d_b),
            vecmem::get_data(d_c),
            scalar);
        #else
            vecpar::omp::parallel_map(array_size,
                                      [&] (int idx) {
                        a->at(idx) = b->at(idx) + scalar * c->at(idx); });
        #endif
    #endif
}

template <class T>
void VecparStream<T>::nstream()
{
    const T scalar = startScalar;

    #if defined(SINGLE_SOURCE)
        vecpar_nstream<T> algorithm;
        vecpar::parallel_algorithm(algorithm, memoryResource, *a, *b, *c, scalar);
    #else
    #ifdef VECPAR_GPU
        vecpar::cuda::parallel_map(
                array_size,
                [=] __device__ (int idx,
                vecmem::data::vector_view<T> &a_view,
                vecmem::data::vector_view<T> &b_view,
                vecmem::data::vector_view<T> &c_view,
                T dscalar) {
            vecmem::device_vector<T> da(a_view);
            vecmem::device_vector<T> db(b_view);
            vecmem::device_vector<T> dc(c_view);
             da[idx] += db[idx] + dscalar * dc[idx];
           },
        vecmem::get_data(d_a),
        vecmem::get_data(d_b),
        vecmem::get_data(d_c),
        scalar);
    #else
        vecpar::omp::parallel_map(array_size,
                              [&] (int idx)  { a->at(idx) = b->at(idx) + scalar * c->at(idx);});
    #endif
#endif
}

template <class T>
T VecparStream<T>::dot()
{
    T* sum = new T();
    *sum = 0.0;
#if defined(SINGLE_SOURCE)
    vecpar_dot<T> algorithm;
    *sum = vecpar::parallel_algorithm(algorithm, memoryResource, *a, *b);
#else
    #ifdef VECPAR_GPU
        T* dsum;
        cudaMalloc(&dsum, sizeof(T));
        vecpar::cuda::offload_reduce(array_size,
                                     [=](int* lock, int size, T* dsum,
                                             vecmem::data::vector_view<T> a_view,
                                             vecmem::data::vector_view<T> b_view) {
                                         vecmem::device_vector<T> da(a_view);
                                         vecmem::device_vector<T> db(b_view);
                                         size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                                         if (idx >= size)
                                             return;
                                         atomicAdd(dsum, (da[idx]*db[idx]));
            }, array_size, dsum,
            vecmem::get_data(d_a),
            vecmem::get_data(d_b));
        cudaMemcpy(sum, dsum, sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(dsum);
    #else
        vecmem::vector<T> result(array_size, &memoryResource);
        vecpar::omp::parallel_map(array_size,
                                  [&] (int idx) { result.at(idx) = a->at(idx) * b->at(idx); });
        vecpar::omp::parallel_reduce(array_size, sum,
                              [&] (T* sum, T& value)  { *sum += value; }, result);
    #endif
#endif
    return *sum;
}

void listDevices(void)
{
#ifdef VECPAR_GPU
    std::cout << "Not implemented yet" << std::endl;
#else
    std::cout << "0: CPU" << std::endl;
#endif
}

std::string getDeviceName(const int)
{
    return std::string("Device name unavailable");
}

std::string getDeviceDriver(const int)
{
    return std::string("Device driver unavailable");
}

template class VecparStream<float>;
template class VecparStream<double>;