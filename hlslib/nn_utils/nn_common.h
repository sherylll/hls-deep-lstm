// author: Yuxi Sun
// Based on nnet_utils from HLS4ML
//

#ifndef NN_COMMON_H_
#define NN_COMMON_H_

#include <hls_math.h>

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

// workaround to pipeline fp accumulation; n_steps = sqrt(arr_size)
// template <class data_T>
// void step10_sum(data_T *arr, data_T *sum, int len)
// {
// //#pragma HLS INLINE
// #pragma HLS PIPELINE

//     data_T temp = 0;
//     int reminder = len % 10;
//     int len1 = len - reminder;

//     for (int i = 9; i < len1; i += 10)
//     {
//         temp += arr[i] + arr[i - 1] + arr[i - 2] + arr[i - 3] + arr[i - 4] + arr[i - 5] + arr[i - 6] + arr[i - 7] + arr[i - 8] + arr[i - 9];
//     }

//     data_T temp_r = 0;
//     for (int i = 0; i < reminder; i++)
//     {
//         temp_r += arr[len1 + i];
//     }
//     *sum = temp + temp_r;
// }

struct fc_config
{
    // Internal data type definitions
    typedef float bias_t;
    typedef float weight_t;
    typedef float accum_t;

    // Layer Sizes
    static const unsigned n_in = 10;
    static const unsigned n_out = 10;
};

// fully connected layer
template <class data_T, class res_T, typename CONFIG_T>
void fc(typename CONFIG_T::weight_t mat[CONFIG_T::n_out][CONFIG_T::n_in], data_T vec[CONFIG_T::n_in],
		typename CONFIG_T::bias_t bias[CONFIG_T::n_out], res_T res[CONFIG_T::n_out])
{
	typename CONFIG_T::accum_t accum = 0;
	typename CONFIG_T::accum_t temp;

    for (int i = 0; i < CONFIG_T::n_out; i++)
    {
#pragma HLS PIPELINE
    	accum = 0;
        for (int j = 0; j < CONFIG_T::n_in; j++)
        {
        	temp= mat[i][j] * vec[j];
            accum += temp;
        }
        res[i] = accum + bias[i];
    }
}

template <class weight_T, class data_T, class res_T, unsigned int col, unsigned int row>
void mat_vec_mul(weight_T mat[row][col], data_T vec[col], res_T res[row])
{
#pragma HLS inline off
    res_T dot_product = 0;

    for (int i = 0; i < row; i++)
    {
#pragma HLS PIPELINE
    	dot_product = 0;
        for (int j = 0; j < col; j++)
        {
        	res_T temp = mat[i][j] * vec[j];
// #pragma HLS RESOURCE variable=temp core=Mul_LUT
        	dot_product = temp + dot_product;
        }
        res[i] = dot_product;
    }
}

template <class weight_T, class data_T, class res_T, unsigned int col, unsigned int row>
void mat_vec_mul_4(weight_T mat[row][col], weight_T mat1[row][col], weight_T mat2[row][col], weight_T mat3[row][col], data_T vec[col], res_T res[row]
																																				 , res_T res1[row], res_T res2[row], res_T res3[row])
{
#pragma HLS inline off
    res_T dot_product,dot_product1,dot_product2,dot_product3;
    res_T temp,temp1,temp2,temp3;

    for (int i = 0; i < row; i++)
    {
#pragma HLS PIPELINE
#pragma HLS unroll factor=2
    	dot_product = 0; dot_product1 = 0; dot_product2 = 0; dot_product3 = 0;
        for (int j = 0; j < col; j++)
        {
        	data_t x = vec[j];
        	temp = mat[i][j] * x;
        	dot_product = temp + dot_product;

        	temp1 = mat1[i][j] * x;
			dot_product1 = temp1 + dot_product1;

			temp2 = mat2[i][j] * x;
			dot_product2 = temp2 + dot_product2;

			temp3 = mat3[i][j] * x;
			dot_product3 = temp3 + dot_product3;
        }
        res[i] = dot_product; res1[i] = dot_product1;res2[i] = dot_product2;res3[i] = dot_product3;
    }
}
} // namespace nn

#endif
