// by HLS4ML
//

#ifndef NNET_ACTIVATION_H_
#define NNET_ACTIVATION_H_

#include "nn_common.h"

namespace nn
{
// this implementation is not suitable for fixed-point numbers

//template <class data_T, class max_T, typename CONFIG_T>
//void softmax(data_T data[CONFIG_T::n_in], data_T res[CONFIG_T::n_in])
//{
//#pragma HLS inline
//
//    max_T y_sum = 0;
//    max_T y_exp[CONFIG_T::n_in];
//#pragma HLS ARRAY_PARTITION variable = y_exp complete dim = 1
//    for (int i = 0; i < CONFIG_T::n_in; i++)
//    {
//#pragma HLS PIPELINE
//        y_exp[i] = hls::exp(data[i]);
//    }
//
//    for (int i = 0; i < CONFIG_T::n_in; i++)
//#pragma HLS PIPELINE
//        y_sum += y_exp[i];
//
//    data_T y_sum_inv = (data_T)1.0 / y_sum; // For  ap_[u]fixed  types, the fraction is no greater than that of the dividend.
//    for (int i = 0; i < CONFIG_T::n_in; i++)
//    {
//#pragma HLS PIPELINE
//        res[i] = y_exp[i] * y_sum_inv; // write results
//    }
//}

// *************************************************
//       TanH Activation
// *************************************************

template<typename CONFIG_T, int N_TABLE>
void init_tanh_table(typename CONFIG_T::table_t table_out[N_TABLE])
{
    // Implement tanh lookup
    for (int ii = 0; ii < N_TABLE; ii++) {
        // First, convert from table index to X-value (signed 8-bit, range -4 to +4)
        float in_val = 2*4.0*(ii-float(N_TABLE)/2.0)/float(N_TABLE);
        // Next, compute lookup table function
        typename CONFIG_T::table_t real_val = tanh(in_val);
        //std::cout << "Tanh:  Lookup table Index: " <<  ii<< " In Value: " << in_val << " Result: " << real_val << std::endl;
        table_out[ii] = real_val;
    }
}


template<class data_T, class res_T, typename CONFIG_T>
void  tanh(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in])
{
    // Initialize the lookup table
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::table_t tanh_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::table_t tanh_table[CONFIG_T::table_size];
#endif
    if (!initialized) {
        init_tanh_table<CONFIG_T, CONFIG_T::table_size>(tanh_table);
        initialized = true;
    }

#pragma HLS PIPELINE
    // Index into the lookup table based on data
    int data_round;
    int index;
    for (int ii=0; ii<CONFIG_T::n_in; ii++) {
        data_round = data[ii]*CONFIG_T::table_size/8;
        index = data_round + 4*CONFIG_T::table_size/8;
        //std::cout << "Input: "  << data[ii] << " Round: " << data_round << " Index: " << index << std::endl;
        if (index < 0)   index = 0;
        if (index > CONFIG_T::table_size-1) index = CONFIG_T::table_size-1;
        res[ii] = (res_T) tanh_table[index];
    }
}

// *************************************************
//       Softmax Activation
// *************************************************
inline float exp_fcn_float(float input)
{
    return std::exp(input);
}

template <typename CONFIG_T, int N_TABLE>
void init_exp_table(typename CONFIG_T::table_t table_out[N_TABLE])
{
    for (int ii = 0; ii < N_TABLE; ii++)
    {
        // First, convert from table index to X-value (signed 8-bit, range -8 to +8)
        float in_val = 2 * 8.0 * (ii - float(N_TABLE) / 2.0) / float(N_TABLE);
        // Next, compute lookup table function
        typename CONFIG_T::table_t real_val = exp_fcn_float(in_val);
        //std::cout << "Lookup table In Value: " << in_val << " Result: " << real_val << std::endl;
        table_out[ii] = real_val;
    }
}

template <typename CONFIG_T, int N_TABLE>
void init_invert_table(typename CONFIG_T::table_t table_out[N_TABLE])
{
    // Inversion function:
    //   result = 1/x
    for (int ii = 0; ii < N_TABLE; ii++)
    {
        // First, convert from table index to X-value (signed 8-bit, range 0 to +64)
        float in_val = 64.0 * ii / float(N_TABLE);
        // Next, compute lookup table function
        if (in_val > 0.0)
            table_out[ii] = 1.0 / in_val;
        else
            table_out[ii] = 0.0;
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void softmax(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in])
{
    // Initialize the lookup table
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::table_t exp_table[CONFIG_T::table_size];
    typename CONFIG_T::table_t invert_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::table_t exp_table[CONFIG_T::table_size];
    static typename CONFIG_T::table_t invert_table[CONFIG_T::table_size];
#endif
    if (!initialized)
    {
        init_exp_table<CONFIG_T, CONFIG_T::table_size>(exp_table);
        init_invert_table<CONFIG_T, CONFIG_T::table_size>(invert_table);
        initialized = true;
    }

#pragma HLS PIPELINE II = 20 // II=1 consumes too much DSP

    // Index into the lookup table based on data for exponentials
    typename CONFIG_T::table_t exp_res[CONFIG_T::n_in]; // different, independent, fixed point precision
    typename CONFIG_T::table_t exp_diff_res;            // different, independent, fixed point precision
    data_T data_cache[CONFIG_T::n_in];
    int data_round;
    int index;
    for (int ii = 0; ii < CONFIG_T::n_in; ii++)
    {
        data_cache[ii] = data[ii];
        exp_res[ii] = 0;
    }
    for (int ii = 0; ii < CONFIG_T::n_in; ii++)
    {
        for (int jj = 0; jj < CONFIG_T::n_in; jj++)
        {
            if (ii == jj)
                exp_diff_res = 1;
            else
            {
                data_round = (data_cache[jj] - data_cache[ii]) * CONFIG_T::table_size / 16;
                index = data_round + 8 * CONFIG_T::table_size / 16;
                if (index < 0)
                    index = 0;
                if (index > CONFIG_T::table_size - 1)
                    index = CONFIG_T::table_size - 1;
                exp_diff_res = exp_table[index];
            }
            exp_res[ii] += exp_diff_res;
        }
    }

    //Second loop to invert
    for (int ii = 0; ii < CONFIG_T::n_in; ii++)
    {
        int exp_res_index = exp_res[ii] * CONFIG_T::table_size / 64;
        if (exp_res_index < 0)
            exp_res_index = 0;
        if (exp_res_index > CONFIG_T::table_size - 1)
            exp_res_index = CONFIG_T::table_size - 1;
        res[ii] = (res_T)invert_table[exp_res_index];
    }
}

// *************************************************
//       Sigmoid Activation
// *************************************************

template <class data_T, class res_T, typename CONFIG_T>
void hard_sigmoid(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in])
{
//#pragma HLS PIPELINE
    data_T datareg;
    data_T slope = (data_T)0.2;
    data_T shift = (data_T)0.5;
    for (int ii = 0; ii < CONFIG_T::n_in; ii++)
    {
#pragma HLS unroll factor=CONFIG_T::unroll_factor
    	data_T temp=slope * data[ii];

        datareg = temp + shift;
        if (datareg > 1)
            datareg = 1;
        else if (datareg < 0)
            datareg = 0;
        res[ii] = datareg;
    }
}

inline float sigmoid_fcn_float(float input)
{
    return 1.0 / (1 + hls::exp(-input));
}

template <typename CONFIG_T, int N_TABLE>
void init_sigmoid_table(typename CONFIG_T::table_t table_out[N_TABLE])
{
    // Default logistic sigmoid function:
    //   result = 1/(1+e^(-x))
    for (int ii = 0; ii < N_TABLE; ii++)
    {
        // First, convert from table index to X-value (signed 8-bit, range -8 to +8)
        float in_val = 2 * 8.0 * (ii - float(N_TABLE) / 2.0) / float(N_TABLE);
        // Next, compute lookup table function
        typename CONFIG_T::table_t real_val = sigmoid_fcn_float(in_val);
        //std::cout << "Lookup table In Value: " << in_val << " Result: " << real_val << std::endl;
        table_out[ii] = real_val;
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void sigmoid(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in])
{
    // Initialize the lookup table
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::table_t sigmoid_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::table_t sigmoid_table[CONFIG_T::table_size];
#endif
    if (!initialized)
    {
        init_sigmoid_table<CONFIG_T, CONFIG_T::table_size>(sigmoid_table);
        initialized = true;
    }
#pragma HLS PIPELINE
    // Index into the lookup table based on data
    int data_round;
    int index;
    for (int ii = 0; ii < CONFIG_T::n_in; ii++)
    {
        data_round = data[ii] * CONFIG_T::table_size / 16;
        index = data_round + 8 * CONFIG_T::table_size / 16;
        if (index < 0)
            index = 0;
        if (index > CONFIG_T::table_size - 1)
            index = CONFIG_T::table_size - 1;
        res[ii] = (res_T)sigmoid_table[index];
    }
}
} // namespace nn

#endif
