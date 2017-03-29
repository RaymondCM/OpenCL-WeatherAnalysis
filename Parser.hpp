#include <fstream>
#include "SimpleTimer.hpp"

namespace Parse {
	void NumericData(char data[], std::vector<int>& dest) {
		dest.push_back(::atoi(data));
	};

	void NumericData(char data[], std::vector<float>& dest) {
		dest.push_back(::atof(data));
	};

	template<typename T>
	void FileEOL(std::string file_path, std::vector<T>& destination) {
		//Open input stream to file
		//Seek to the end of the file and get the position of the last char
		std::ifstream input_file(file_path, std::ios::in | std::ios::binary | std::ios::ate);

		//Get Size of File
		std::size_t size = input_file.tellg();

		//Seek back to the start of the file
		input_file.seekg(0, std::ios_base::beg);

		//Read into char buffer with size of file
		char * file_contents = new char[size];
		input_file.read(&file_contents[0], size);

		//Close the file
		input_file.close();
		for (unsigned int i = 0; i < size; ++i) {
			if (file_contents[i] == '\n') {
				char word[7] = {"     \0"};
				int j = i, counter = 6;

				while(file_contents[j] != ' ')
					*(word + counter--) = file_contents[j--];

				Parse::NumericData(word, destination);
			}
		};
	};

	template<typename T>
	void File(std::string file_path, std::vector<T>& destination) {
        SimpleTimer t;
		t.Tic();
		Parse::FileEOL(file_path, destination);
        std::cout << "File Parsed in " << t.Toc() / 1000000 << "ms" << std::endl;
	};

	template<typename T>
	std::vector<T> File(std::string file_path) {
		std::vector<T> data;
		File(file_path, data);
		return data;
	};
}