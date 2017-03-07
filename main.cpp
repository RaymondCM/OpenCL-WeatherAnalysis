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

#include "Utils.hpp"

bool ParseFile(std::string file_path, std::vector<double>& dest) {
	//Open input stream to file
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
			dest.push_back(strtod(word, NULL));
		}
	};

	return true;
};

int main(int argc, char **argv) {

	std::string file_path(".");

	#ifdef PROJECT_ROOT
	file_path = PROJECT_ROOT;
	#endif

	file_path += "/data/temp_lincolnshire.txt";

	std::vector<double> temperature;
	ParseFile(file_path, temperature);

	return 0;
}
