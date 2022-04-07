#include <ipudef.h>
#include <poplar/Vertex.hpp>
#include <print.h>
using namespace poplar;

#define UNROLL 8

template <typename T, typename V>
class InitKernel : public Vertex
{

public:
    Output<Vector<T>> a, b, c;
    unsigned size;
    Input<float> initA, initB, initC;

    bool compute()
    {
        for (auto i = 0u; i < size; i++)
        {
            a[i] = initA;
            b[i] = initB;
            c[i] = initC;
        }
        return true;
    }
};

template class InitKernel<float, float>;
template class InitKernel<half, half>;
template class InitKernel<float, float2>;
template class InitKernel<half, half4>;

template <typename T, typename V>
class CopyKernel : public Vertex
{

public:
    Input<Vector<T, VectorLayout::ONE_PTR, 8>> a;
    Output<Vector<T, VectorLayout::ONE_PTR, 8>> c;
    unsigned size;

    inline void doCopy(const V *__restrict src, V *__restrict dst, const unsigned size)
    {
#pragma unroll UNROLL
        for (auto i = 0u; i < size; i++)
        {
            dst[i] = src[i];
        }
    }

    bool compute()
    {
        doCopy(reinterpret_cast<V *>(&a[0]), reinterpret_cast<V *>(&c[0]), size * sizeof(T) / sizeof(V));
        return true;
    }
};

template class CopyKernel<float, float>;
template class CopyKernel<half, half>;
template class CopyKernel<float, float2>;
template class CopyKernel<half, half4>;

template <typename T, typename V>
class MulKernel : public Vertex
{

public:
    Input<Vector<T, VectorLayout::ONE_PTR, 8>> c;
    Output<Vector<T, VectorLayout::ONE_PTR, 8>> b;
    unsigned size;
    float alpha;

    inline void doMul(const V *__restrict src, V *__restrict dst, const float alpha, const unsigned size)
    {
#pragma unroll UNROLL
        for (auto i = 0u; i < size; i++)
        {
            dst[i] = alpha * src[i];
        }
    }

    bool compute()
    {
        doMul(reinterpret_cast<V *>(&c[0]), reinterpret_cast<V *>(&b[0]), alpha, size * sizeof(T) / sizeof(V));
        return true;
    }
};

template class MulKernel<float, float>;
template class MulKernel<half, half>;
template class MulKernel<float, float2>;

template <>
class MulKernel<half, half4> : public Vertex
{

public:
    Input<Vector<half, VectorLayout::ONE_PTR, 8>> c;
    Output<Vector<half, VectorLayout::ONE_PTR, 8>> b;
    unsigned size;
    float alpha;

    inline void doMul(const half4 *__restrict src, half4 *__restrict dst, const float alpha, const unsigned size)
    {
        half _alpha = (half)alpha;
#pragma unroll UNROLL
        for (auto i = 0u; i < size; i++)
        {
            dst[i] = _alpha * src[i];
        }
    }

    bool compute()
    {
        doMul(reinterpret_cast<half4 *>(&c[0]), reinterpret_cast<half4 *>(&b[0]), alpha, size / 4);
        return true;
    }
};

template <typename T, typename V>
class AddKernel : public Vertex
{

public:
    Input<Vector<T, VectorLayout::ONE_PTR, 8>> b;
    Input<Vector<T, VectorLayout::ONE_PTR, 8>> a;
    Output<Vector<T, VectorLayout::ONE_PTR, 8>> c;
    unsigned size;

    inline void doAdd(const V *__restrict a, const V *__restrict b, V *__restrict c, const unsigned size)
    {
#pragma unroll UNROLL
        for (auto i = 0u; i < size; i++)
        {
            c[i] = a[i] + b[i];
        }
    }

    bool compute()
    {
        doAdd(reinterpret_cast<V *>(&a[0]), reinterpret_cast<V *>(&b[0]), reinterpret_cast<V *>(&c[0]), size * sizeof(T) / sizeof(V));
        return true;
    }
};

template class AddKernel<float, float>;
template class AddKernel<half, half>;
template class AddKernel<float, float2>;
template class AddKernel<half, half4>;

template <typename T, typename V>
class TriadKernel : public Vertex
{

public:
    Input<Vector<T>> b;
    Input<Vector<T>> c;
    Output<Vector<T>> a;
    float alpha;
    unsigned size;

    inline void doTriad(V *__restrict a, const V *__restrict b, const V *__restrict c, const float alpha, const unsigned size)
    {
#pragma unroll UNROLL
        for (auto i = 0u; i < size; i++)
        {
            a[i] = b[i] + alpha * c[i];
        }
    }

    bool compute()
    {
        doTriad(reinterpret_cast<V *>(&a[0]), reinterpret_cast<V *>(&b[0]), reinterpret_cast<V *>(&c[0]), alpha, size * sizeof(T) / sizeof(V));
        return true;
    }
};

template class TriadKernel<float, float>;
template class TriadKernel<half, half>;
template <>
class TriadKernel<float, float2> : public Vertex
{

public:
    Input<Vector<float, VectorLayout::ONE_PTR, 8>> b;
    Input<Vector<float, VectorLayout::ONE_PTR, 8>> c;
    Output<Vector<float, VectorLayout::ONE_PTR, 8>> a;
    float alpha;
    unsigned size;

    inline void doTriad(float2 *__restrict a, const float2 *__restrict b, const float2 *__restrict c, const float alpha, const unsigned size)
    {
#pragma unroll UNROLL
        for (auto i = 0u; i < size; i++)
        {
            a[i] = b[i] + alpha * c[i];
        }
    }

    bool compute()
    {
        doTriad(reinterpret_cast<float2 *>(&a[0]), reinterpret_cast<float2 *>(&b[0]), reinterpret_cast<float2 *>(&c[0]), alpha, size / 2);
        return true;
    }
};
template <>
class TriadKernel<half, half4> : public Vertex
{

public:
    Input<Vector<half, VectorLayout::ONE_PTR, 8>> b;
    Input<Vector<half, VectorLayout::ONE_PTR, 8>> c;
    Output<Vector<half, VectorLayout::ONE_PTR, 8>> a;
    float alpha;
    unsigned size;

    inline void doTriad(half4 *__restrict a, const half4 *__restrict b, const half4 *__restrict c, const float alpha, const unsigned size)
    {
#pragma unroll UNROLL
        for (auto i = 0u; i < size; i++)
        {
            a[i] = b[i] + (half)alpha * c[i];
        }
    }

    bool compute()
    {
        doTriad(reinterpret_cast<half4 *>(&a[0]), reinterpret_cast<half4 *>(&b[0]), reinterpret_cast<half4 *>(&c[0]), alpha, size / 4);
        return true;
    }
};

template <typename T, typename V>
class DotProdKernel : public Vertex
{

public:
    Input<Vector<T>> a;
    Input<Vector<T>> b;
    Output<float> sum;
    unsigned size;

    inline auto doDotProd(const V *__restrict a, const V *__restrict b, const unsigned size) -> float
    {
        float tmp = 0.f;
#pragma unroll UNROLL
        for (auto i = 0u; i < size; i++)
        {
            tmp += a[i] * b[i];
        }
        return tmp;
    }

    bool compute()
    {
        *sum = doDotProd(reinterpret_cast<V *>(&a[0]), reinterpret_cast<V *>(&b[0]), size * sizeof(T) / sizeof(V));
        return true;
    }
};

template class DotProdKernel<float, float>;
template class DotProdKernel<half, half>;
template <>
class DotProdKernel<float, float2> : public Vertex
{

public:
    Input<Vector<float, VectorLayout::ONE_PTR, 8>> a;
    Input<Vector<float, VectorLayout::ONE_PTR, 8>> b;
    Output<float> sum;
    unsigned size;

    inline auto doDotProd(const float2 *__restrict a, const float2 *__restrict b, const unsigned size) -> float
    {
        float2 tmp = {0.f, 0.f};
#pragma unroll UNROLL
        for (auto i = 0u; i < size; i++)
        {
            tmp += a[i] * b[i];
        }
        return (float)tmp[0] + tmp[1];
    }
    bool compute()
    {
        *sum = doDotProd(reinterpret_cast<float2 *>(&a[0]), reinterpret_cast<float2 *>(&b[0]), size / 2);
        return true;
    }
};
template <>
class DotProdKernel<half, half4> : public Vertex
{

public:
    Input<Vector<half, VectorLayout::ONE_PTR, 8>> a;
    Input<Vector<half, VectorLayout::ONE_PTR, 8>> b;
    Output<float> sum;
    unsigned size;
    inline auto doDotProd(const half4 *__restrict a, const half4 *__restrict b, const unsigned size) -> float
    {
        half4 tmp = {0, 0, 0, 0};
#pragma unroll UNROLL
        for (auto i = 0u; i < size; i++)
        {
            tmp += a[i] * b[i];
        }
        return (float)tmp[0] + tmp[1] + tmp[2] + tmp[3];
    }

    bool compute()
    {
        *sum = doDotProd(reinterpret_cast<half4 *>(&a[0]), reinterpret_cast<half4 *>(&b[0]), size) / 4;
        return true;
    }
};

class ReduceSum : public Vertex
{
public:
    Input<Vector<float>> partialSums;
    Output<float> sum;
    unsigned size;

    inline auto doReduceSum(const float *__restrict partials, const unsigned size) -> float
    {
        float tmp = 0.f;
#pragma unroll UNROLL
        for (auto i = 0u; i < size; i++)
        {
            tmp += partials[i];
        }
        return tmp;
    }

    bool compute()
    {
        *sum = doReduceSum(reinterpret_cast<float *>(&partialSums[0]), size);
        return true;
    }
};
