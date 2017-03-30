//Raymond Kirk - 14474219@students.lincoln.ac.uk

#ifndef ASSIGNMENTONE_SIMPLETIMER_H
#define ASSIGNMENTONE_SIMPLETIMER_H

#include <chrono>

//Timer class to accuratly record benchmarks for host code.
// Uses matlab style syntax for readability.
class SimpleTimer {
public:
    void Tic(){
        start_t = std::chrono::steady_clock::now();
    };

    long long int Toc(){
        return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start_t).count();
    };
private:
    std::chrono::steady_clock::time_point start_t;
};

#endif //ASSIGNMENTONE_SIMPLETIMER_H
