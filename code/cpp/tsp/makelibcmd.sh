#!/bin/bash

for i in {3..7}; 
do
	sed -i "s/PYBIND11_MODULE(libtspenvv2, m)/PYBIND11_MODULE(libtspenvv2o$i, m)/g" src/libtspenv.cpp
	
	makelib=$(echo "g++ -m64 -O3 -DIL_STD -DNOBJS=$i -Wall -shared -std=c++17 -fPIC -I./include $(python3 -m pybind11 --includes) src/libtspenv.cpp src/tspenv.cpp src/dd/*.cpp src/instance/*.cpp src/util/*.cpp $(python3-config --ldflags) -lm -ldl -pthread -o libtspenvv2o$i.cpython-38-x86_64-linux-gnu.so")
	
	eval $makelib

	sed -i "s/PYBIND11_MODULE(libtspenvv2o$i, m)/PYBIND11_MODULE(libtspenvv2, m)/g" src/libtspenv.cpp
done
