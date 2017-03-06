#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <stdio.h>
#include <string>

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "Utils.h"

int main(int argc, char **argv) {

	char * file_path = "../../ParallelComputing/Tutorial 3/temp_lincolnshire_short.txt";
	std::ifstream input_file(file_path);

	//Seek to the end of the file and get the position of the last char
	input_file.seekg(0, std::ios_base::end);
	std::size_t size = input_file.tellg();

	//Seek back to the start of the file
	input_file.seekg(0, std::ios_base::beg);

	//Read into char buffer with size of file
	char * file_contents = new char[size];
	input_file.read(&file_contents[0], size);

	//Close the file
	input_file.close();

	//Parse the file by keeping track of the last space before \n
	long space_pos = 0;

	//Final vector of values
	std::vector <double> values;

	for (long i = 0; i < size; ++i) {
		char c = file_contents[i];

		if (c == ' ') {
			//+1 to i to itterate after last space
			space_pos = i + 1;
		}
		else if (c == '\n') {
			int len = i - space_pos, index = 0;
			
			//Allocate buffer for word between space_pos and \n
			char * word = new char[len];
			word[len] = '\0';

			//Get every char between last space and \n
			for (int j = space_pos; j < i; j++) {
				word[index++] = file_contents[j];
			}

			//Parse word to double (higher precison)
			values.push_back(strtod(word, NULL));
		}
	}

	return 0;
}
