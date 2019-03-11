// author: Yuxi Sun
// Based on nnet_utils from HLS4ML
//

#ifndef NN_COMMON_H_
#define NN_COMMON_H_

#include <hls_math.h>
//#include <cmath>

namespace nn
{
enum activ_type
{
    activ_relu = 0,
    activ_sigmoid,
    activ_tanh,
    activ_softmax,
    activ_hard_sigmoid
};

struct fc_config
{
    static const unsigned row = 2;
    static const unsigned col = 2;
};

// workaround to get fp accumulation pipelined; steps of 10 works best for total size = 100
template <class data_T>
void step10_sum(data_T *arr, data_T *sum, int len)
{
//#pragma HLS INLINE
#pragma HLS PIPELINE

    data_T temp = 0;
    int reminder = len % 10;
    int len1 = len - reminder;

    for (int i = 9; i < len1; i += 10)
    {
        temp += arr[i] + arr[i - 1] + arr[i - 2] + arr[i - 3] + arr[i - 4] + arr[i - 5] + arr[i - 6] + arr[i - 7] + arr[i - 8] + arr[i - 9];
    }

    data_T temp_r = 0;
    for (int i = 0; i < reminder; i++)
    {
        temp_r += arr[len1 + i];
    }
    *sum = temp + temp_r;
}

// computes Wx + b (untransposed). Improved for FP operations. When using this, framework (Keras/TF) weights
// need to be transposed
template <class data_T, unsigned int col, unsigned int row>
void fc(data_T mat[row][col], data_T vec[col],
        data_T bias[row], data_T res[row])
{
    data_T Wly_dot_h_sum = 0;
    data_T Wly_dot_h[col];

#pragma HLS ARRAY_PARTITION variable = Wly_dot_h cyclic factor = 10 dim = 1 // TODO make this configurable
    for (int i = 0; i < row; i++)
    {
#pragma HLS PIPELINE
        Wly_dot_h_sum = 0;
        for (int j = 0; j < col; j++)
        {
            Wly_dot_h[j] = mat[i][j] * vec[j];
        }

        // used due to arr len = 100, optimal #steps should be sqrt(arr_len)
        nn::step10_sum<data_T>(Wly_dot_h, &Wly_dot_h_sum, col);
        res[i] = Wly_dot_h_sum + bias[i];
    }
}

template <class data_T, class res_T, unsigned int col, unsigned int row>
void fc_no_b(data_T mat[row][col], data_T vec[col], res_T res[row])
{
    res_T Wly_dot_h_sum = 0;
//#pragma HLS ARRAY_PARTITION variable = Wly_dot_h complete dim = 1 // TODO make this configurable
    for (int i = 0; i < row; i++)
    {
#pragma HLS PIPELINE
        Wly_dot_h_sum = 0;
        for (int j = 0; j < col; j++)
        {
        	data_T temp = mat[i][j] * vec[j];
// #pragma HLS RESOURCE variable=temp core=Mul_LUT
        	Wly_dot_h_sum = temp+Wly_dot_h_sum;
        }
        res[i] = Wly_dot_h_sum;
    }
}
} // namespace nn

#endif
