#include "parameter.h"
#include <string>
#include <cstring>
#include <iostream>
#include <fstream>

 #define N_LOOP 98
 #define N_INPUTS 16
 #define TEST_SIZE 10

 void lstmp(data_t data[N_LOOP][N_INPUTS],
 		ap_uint<4> *res);
 int read_to_array(char *path, data_t x_test[N_LOOP][N_INPUTS], int *y_test)
 {
 	std::ifstream inFile;
 	inFile.open(path);
 	if (!inFile)
 		return -1;
 	if (inFile.get() == '#')
 		inFile >> *y_test;
 	for (int i = 0; i < N_LOOP; i++)
 	{
 		for (int j = 0; j < N_INPUTS; j++)
 		{
 			inFile >> x_test[i][j];
 		}
 	}
 	inFile.close();
 	return 0;
 }
 int main()
 {
 	char x_str[10] = "";
 	char path_cstr[100];

 	data_t x_test[N_LOOP][N_INPUTS];
 	int y_test, counter;
 	ap_uint<4> y_pred;
 	for (int im=0; im < TEST_SIZE; im ++){
 		sprintf(x_str, "%d.txt", im);
 		std::string image_path = "/home/asap2/hikari/vivado_prj_quant_lstm/test_speech_commands/test_data/";
 		image_path += std::string(x_str);
 		strcpy(path_cstr, image_path.c_str());
 		if (read_to_array(path_cstr, x_test, &y_test) == 0){
 			lstmp(x_test, &y_pred);
 			std::cout << im << " " << (y_pred == y_test) << std::endl;
 			if (y_pred == y_test)
 				counter++;
 		}
 		else
 			std::cout << "failed to read file" << std::endl;
 	}
 	std::cout << counter;
 }

// debugging
//#include "parameter.h"
//#include "../hlslib/fic_utils/fic_packet.h"
//
//#include "weights/lstm0/Wi.h"
//#include "weights/lstm0/Wf.h"
//#include "weights/lstm0/Wc.h"
//#include "weights/lstm0/Wo.h"
//#include "weights/lstm0/Ui.h"
//#include "weights/lstm0/Uf.h"
//#include "weights/lstm0/Uc.h"
//#include "weights/lstm0/Uo.h"
//#include "weights/lstm0/bi.h"
//#include "weights/lstm0/bf.h"
//#include "weights/lstm0/bc.h"
//#include "weights/lstm0/bo.h"
//#include "weights/lstm0/proj.h"
//#include "weights/lstm1/Wi.h"
//#include "weights/lstm1/Wf.h"
//#include "weights/lstm1/Wc.h"
//#include "weights/lstm1/Wo.h"
//#include "weights/lstm1/Ui.h"
//#include "weights/lstm1/Uf.h"
//#include "weights/lstm1/Uc.h"
//#include "weights/lstm1/Uo.h"
//#include "weights/lstm1/bi.h"
//#include "weights/lstm1/bf.h"
//#include "weights/lstm1/bc.h"
//#include "weights/lstm1/bo.h"
//#include "weights/lstm1/proj.h"
//#include "../test_data/mfcc_4.h"
//void lstm0(ap_uint<4> rpi_ctrl,
//		ap_uint<169> h0_packets[N_PACKETS])
//{
//
//    static lstm_t h0_oldstate[N_PROJ] = {0};
//    static lstm_t c0_oldstate[N_STATES] = {0};
//     lstm_t h0_newstate[N_PROJ] = {0};
//     lstm_t c0_newstate[N_STATES] = {0};
//
//    data_t mfcc[N_INPUTS];
//
//    static short timestep = 0;
//    ap_uint<4> ctrl = rpi_ctrl;
//    switch (ctrl){
//        case 0xf:
//            for (int i = 0; i < N_INPUTS; i++)
//                mfcc[i] = mfcc_inputs[timestep * N_INPUTS + i];
//
//            nn::lstmp<data_t, lstm_t, config0, cell_act_config, recurrent_act_config>(mfcc, h0_oldstate, h0_newstate, c0_oldstate, c0_newstate,
//                        Wi, Wf, Wc, Wo, Ui, Uf, Uc, Uo, proj, bi, bf, bc, bo);
//            timestep++;
//            fic::encoder<lstm_t, packet_config>(h0_newstate, h0_packets);
//            break;
//        case 0xe:
//            for (int i = 0; i < N_PROJ; i++)
//            {
//                h0_oldstate[i] = 0; h0_newstate[i] = 0;
//            }
//            for (int i = 0; i < N_STATES; i++)
//            {
//                c0_oldstate[i] = 0; c0_newstate[i] = 0;
//            }
//            timestep = 0;
//            break;
//    }
//}
//
//void lstm1(ap_uint<169> h0_packets[N_PACKETS],
//		lstm_t y_debug[N_PROJ])
//{
//    static lstm_t h1_oldstate[N_PROJ] = {0};
//    static lstm_t c1_oldstate[N_STATES] = {0};
//
//    lstm_t h1_newstate[N_PROJ] = {0}; // creates dependency when data is encoded and sent to the switch, not very useful here...
//    lstm_t c1_newstate[N_STATES] = {0};
//
//    lstm_t h0_states[N_PROJ];
//
//    res_t y[N_OUTPUTS] = {0};
//    static unsigned short timestep1 = 0;
//
//	fic::decoder<lstm_t, packet_config>(h0_packets, h0_states);
//	nn::lstmp<lstm_t, lstm_t, config1, cell_act_config, recurrent_act_config>(h0_states, h1_oldstate, h1_newstate, c1_oldstate, c1_newstate,
//			Wi1, Wf1, Wc1, Wo1, Ui1, Uf1, Uc1, Uo1, proj1, bi1, bf1, bc1, bo1);
//	timestep1 += 1;
//
//    if (timestep1 == N_LOOP) // can HLS properly build an FSM???
//    {
//
//       for (int i = 0; i < N_PROJ; i++) // write out
//       {
//    	   y_debug[i] = h1_newstate[i];
//    	   // 246 247 241 232 28 250 20 235 242 4 245 21 8 9 22 6 15 252 6 247 13 244 32 20 11 23 251 244 252 1 5 239 11 253 38 254 2 26 226 2 250 242 254 16 6 240 218 8 18 252 255 8 232 9 248 251 255 255 12 3 252 239 253 234 223 23 0 246 30 228 25 12 236 252 1 243 0 0 0 0 0 0 0 0 0 0 0 0 48 248 111 164 252 127 0 0 192 248 111 164 252 127 0 0 0 0 0 0 7 0 0 0 0 0 112 164 252 127 0 0 0 247 111 164 252 127 0 0
//       }
//	   for (int i = 0; i < N_PROJ; i++)
//	   {
//		   h1_oldstate[i] = 0; h1_newstate[i] = 0;
//	   }
//	   for (int i = 0; i < N_STATES; i++)
//	   {
//		   c1_oldstate[i] = 0; c1_newstate[i] = 0;
//	   }
//	   timestep1 = 0;
//    }
//    // fic::encoder<lstm_t, packet_config>(h0_newstate, h0_packets);
//}
//
//int main()
//{
//	lstm_t y_debug[N_PROJ];
//	ap_uint<169> h0_packets[N_PACKETS];
//
//	for(int i = 0; i < N_LOOP; i++)
//	{
//		lstm0(0x0f, h0_packets);
//		lstm1(h0_packets,y_debug);
//	}
//	lstm0(0x0e, h0_packets);
//	// lstm1(h0_packets,y_debug);
//
//	for(int i = 0; i < N_PROJ; i++)
//		std::cout << y_debug[i].range() << " ";
//}
