#include "ap_int.h"

namespace fic
{
template<class data_T, typename CONFIG_T>
void encoder(
    data_T input[CONFIG_T::n_data],
    ap_uint<169> output[CONFIG_T::n_packets])
{
#pragma HLS PIPELINE
    ap_uint<169> tmp;
    for (int p=0; p<CONFIG_T::n_packets; p++)
    {
        for (int i = 0; i < CONFIG_T::n_chunks; i++)
        {
            tmp(127 - CONFIG_T::len_data * i, 128-CONFIG_T::len_data * (i+1)) = input[p*CONFIG_T::n_chunks+i].range(); // MSB first
        }
        output[p] = tmp;
    }
}

template<class data_T, typename CONFIG_T>
void decoder(
    ap_uint<169> input[CONFIG_T::n_packets],
    data_T output[CONFIG_T::n_data])
{
#pragma HLS PIPELINE
    ap_uint<169> tmp;

    for (int p=0; p<CONFIG_T::n_packets; p++)
    {
        tmp = input[p];
        for (int i = 0; i < CONFIG_T::n_chunks; i++)
        {
            output[p*CONFIG_T::n_chunks+i].range() = tmp(127 - CONFIG_T::len_data * i, 128-CONFIG_T::len_data * (i+1)); // MSB first
        }        
    }
}
}
