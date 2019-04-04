#include "ap_int.h"

namespace fic
{
template<class data_T, typename CONFIG_T>
void encoder(
    data_T input[CONFIG_T::n_data],
    ap_uint<169> output[CONFIG_T::n_packets],
    ap_uint<16> slot_id = 0) // defaults to slot 0
{
#pragma HLS PIPELINE
    ap_uint<169> tmp;
    for (int p=0; p<CONFIG_T::n_packets; p++)
    {
        for (int i = 0; i < CONFIG_T::n_chunks; i++)
        {
            tmp(127 - CONFIG_T::len_data * i, 128-CONFIG_T::len_data * (i+1)) = input[p*CONFIG_T::n_chunks+i].range(); // MSB first
        }
        tmp(168,153) = slot_id; 
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

/* this encoder tries to occupy all 8 slots
 * NOT SURE IF USEFUL*/
template<class data_T, typename CONFIG_T>
void greedy_encoder(
    data_T input[CONFIG_T::n_data],
    ap_uint<169> output[CONFIG_T::n_packets])
{
#pragma HLS PIPELINE
    ap_uint<169> tmp;
    ap_uint<16> slot_id;
    for (int p=0; p<CONFIG_T::n_packets; p++)
    {
        for (int i = 0; i < CONFIG_T::n_chunks; i++)
        {
            tmp(127 - CONFIG_T::len_data * i, 128-CONFIG_T::len_data * (i+1)) = input[p*CONFIG_T::n_chunks+i].range(); // MSB first
        }
        tmp(168,153) = slot_id;
        output[p] = tmp;
        slot_id ++;
        if (slot_id == 8)
            slot_id = 0;
    }
}
}
