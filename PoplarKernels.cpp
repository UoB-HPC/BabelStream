#include <poplar/Vertex.hpp>
#include <print.h>

using namespace poplar;

class CopyKernel: public Vertex {

public:
    Input<Vector<float>> a; 
    Output<Vector<float>> c; 

    bool compute() {
        for ( auto i = 0u; i < a.size(); i++) {
            c[i] = a[i];
        }
        return true;
    }
};


class MulKernel: public Vertex {

public:
    Input<Vector<float>> c; 
    Output<Vector<float>> b; 
    Input<float> alpha;

    bool compute() {
        for ( auto i = 0u; i < c.size(); i++) {
            b[i] = *alpha * c[i];
        }
        return true;
    }
};


class AddKernel: public Vertex {

public:
    Input<Vector<float>> b; 
    Input<Vector<float>> a; 
    Output<Vector<float>> c; 

    bool compute() {
        for ( auto i = 0u; i < a.size(); i++) {
            c[i] = a[i] + b[i];
        }
        return true;
    }
};


class TriadKernel: public Vertex {

public:
    Input<Vector<float>> b; 
    Input<Vector<float>> c; 
    Output<Vector<float>> a; 
    Input<float> alpha;

    bool compute() {
        for ( auto i = 0u; i < b.size(); i++) {
            a[i] = b[i] + *alpha * c[i];
        }
        return true;
    }
};


class DotProdKernel: public Vertex {

public:
    Input<Vector<float>> a; 
    Input<Vector<float>> b; 
    Output<float> sum; 

    bool compute() {
        *sum = 0.0f;
        for ( auto i = 0u; i < a.size(); i++) {
            *sum = *sum + a[i] * b[i];
        }
        return true;
    }
};

class ReduceSum: public Vertex {
public:
    Input<Vector<float>> partialSums; 
    Output<float> sum; 

    bool compute() {
        *sum = 0.0f;
        for ( auto i = 0u; i < partialSums.size(); i++) {
            *sum += partialSums[i];
        }
        return true;
    }
};


