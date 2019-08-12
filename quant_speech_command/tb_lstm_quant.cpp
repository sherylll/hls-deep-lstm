#include "parameter.h"
#include <string>
#include <cstring>
#include <iostream>
#include <fstream>

#define N_LOOP 98
#define N_INPUTS 16
#define TEST_SIZE 4890

void lstm0(data_t data[N_LOOP][N_INPUTS],
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
	char path_cstr[30];

	data_t x_test[N_LOOP][N_INPUTS];
	int y_test, counter=0;
	ap_uint<4> y_pred;
	for (int im=0; im < TEST_SIZE; im ++){
		sprintf(x_str, "%d.txt", im);
		std::string image_path = "test_data/";
		image_path += std::string(x_str);
		strcpy(path_cstr, image_path.c_str());
		if (read_to_array(path_cstr, x_test, &y_test) == 0){
			lstm0(x_test, &y_pred);
			std::cout << im << " " << (y_pred == y_test) << std::endl;
			if (y_pred == y_test)
				counter++;
		}
		else
			std::cout << "failed to read file" << std::endl;
	}
	std::cout << counter;
}
