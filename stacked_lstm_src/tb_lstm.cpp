#include "firmware/lstm.h"
#include <iostream>

#define IMAGE_WIDTH 28

int max_likelihood(data_t y[N_OUTPUTS])
{
	int i_likely = 0;
	data_t y_max = 0;
	for (int i = 0; i < N_OUTPUTS; i++)
	{
		if (y[i] > y_max)
		{
			y_max = y[i];
			i_likely = i;
		}
	}
	std::cout << y[i_likely] << "  "<<std::endl;
	return i_likely;
}
//
//int read_to_array(char *path, data_t array[IMAGE_WIDTH][IMAGE_WIDTH])
//{
//	std::ifstream inFile;
//	inFile.open(path);
//	if (!inFile)
//		return -1;
//	for (int i = 0; i < IMAGE_WIDTH; i++)
//	{
//		for (int j = 0; j < IMAGE_WIDTH; j++)
//		{
//			inFile >> array[i][j];
//		}
//	}
//	inFile.close();
//	return 0;
//}

int main()
{
	data_t probs[N_OUTPUTS];

	bidirectional_lstm(0x0f,probs);
	std::cout << max_likelihood(probs);
}
