set -x
nvcc -O3 -arch=sm_86 -use_fast_math "$1" -o test -std=c++17 -lcudart -lstdc++ -ccbin gcc-12
./test