#include "parameter.h"
#include <iostream>
#include <fstream>
#define IMAGE_WIDTH 28

data_t max_likelihood(data_t y[N_OUTPUTS])
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
	std::cout << y[i_likely] << "  ";
	return i_likely;
}

int read_to_array(char *path, data_t array[IMAGE_WIDTH][IMAGE_WIDTH])
{
	std::ifstream inFile;
	inFile.open(path);
	if (!inFile)
		return -1;
	for (int i = 0; i < IMAGE_WIDTH; i++)
	{
		for (int j = 0; j < IMAGE_WIDTH; j++)
		{
			inFile >> array[i][j];
		}
	}
	inFile.close();
	return 0;
}

int main()
{
	data_t probs[N_OUTPUTS];

	char image_path[20] = "images/4/4.dat"; // add path to images in HLS
	data_t test_image[N_INPUTS][N_INPUTS];

	if (read_to_array(image_path, test_image) == 0)
		std::cout << "read file" << std::endl;
	else
		std::cout << "failed to read file" << std::endl;

	//	struct timeval  tv1, tv2;
	//	gettimeofday(&tv1, NULL);

	myproject(test_image, probs); // timed to be 0.008s=8ms, according to report 0.15 ms when using fp

	//	gettimeofday(&tv2, NULL);
	//	double dt= (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 +
	//	         (double) (tv2.tv_sec - tv1.tv_sec);
	//	std::cout << dt;
	//	return 0;

	std::cout << "4  " << max_likelihood(probs) << std::endl;

	read_to_array("images/0/3.dat", test_image);
	myproject(test_image, probs);
	std::cout << "0  " << max_likelihood(probs) << std::endl;

	read_to_array("images/1/5.dat", test_image);
	myproject(test_image, probs);
	std::cout << "1  " << max_likelihood(probs) << std::endl;

	read_to_array("images/9/7.dat", test_image);
	myproject(test_image, probs);
	std::cout << "9  " << max_likelihood(probs) << std::endl;

	read_to_array("images/9/9.dat", test_image);
	myproject(test_image, probs);
	std::cout << "9  " << max_likelihood(probs) << std::endl;
}