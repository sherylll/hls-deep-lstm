#include "parameter.h"
#include <string>
#include <cstring>
#include <iostream>
#include <fstream>
#define IMAGE_WIDTH 28
#define TEST_SIZE 10000
void lstm0(data_t data[N_LOOP][N_INPUTS],
           data_t res[N_OUTPUTS]);

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
//	std::cout << y[i_likely] << "  ";
	return i_likely;
}

int read_to_array(char *path, data_t x_test[IMAGE_WIDTH][IMAGE_WIDTH], int *y_test)
{
	std::ifstream inFile;
	inFile.open(path);
	if (!inFile)
		return -1;
	if (inFile.get() == '#')
		inFile >> *y_test;
//	std::cout << *y_test;
	for (int i = 0; i < IMAGE_WIDTH; i++)
	{
		for (int j = 0; j < IMAGE_WIDTH; j++)
		{
			inFile >> x_test[i][j];
		}
	}
	inFile.close();
	return 0;
}

int main()
{
	data_t probs[N_OUTPUTS];
	char x_str[10] = "";
	char path_cstr[30];

	data_t x_test[N_INPUTS][N_INPUTS];
	int y_test, counter;

	for (int im=0; im < TEST_SIZE; im ++){
		sprintf(x_str, "%d.txt", im);
		std::string image_path = "test_images/";
		image_path += std::string(x_str);
		strcpy(path_cstr, image_path.c_str());
		if (read_to_array(path_cstr, x_test, &y_test) == 0){
			lstm0(x_test, probs);
			int y_pred = max_likelihood(probs);
			std::cout << im << " " << (y_pred == y_test)<< std::endl;
			if (y_pred == y_test)
				counter++;
		}
		else
			std::cout << "failed to read file" << std::endl;
	}
	std::cout << counter;
}
