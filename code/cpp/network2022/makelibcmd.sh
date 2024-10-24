#!/bin/bash

CPLEX_ROOT_PATH=/opt/ibm/ILOG/CPLEX_Studio1210
ARCH=x86-64_linux

CPOPT_PATH=$CPLEX_ROOT_PATH/cpoptimizer
CPOPT_INC=$CPOPT_PATH/include
CPOPT_LIB=$CPOPT_PATH/lib/$ARCH/static_pic

CONCERT_PATH=$CPLEX_ROOT_PATH/concert
CONCERT_INC=$CONCERT_PATH/include
CONCERT_LIB=$CONCERT_PATH/lib/$ARCH/static_pic

CPLEX_PATH=$CPLEX_ROOT_PATH/cplex
CPLEX_INC=$CPLEX_PATH/include
CPLEX_LIB=$CPLEX_PATH/lib/$ARCH/static_pic

for i in {3..7}
do
	sed -i "s/PYBIND11_MODULE(libbddenv, m)/PYBIND11_MODULE(libbddenvv21o$i, m)/g" src/libbddenv.cpp
	
	makelib=$(echo "g++ -m64 -O3 -DIL_STD -DNOBJS=$i -Wall -shared -std=c++17 -fPIC -I./include -I$CPOPT_INC -I$CONCERT_INC -I$CPLEX_INC $(python3 -m pybind11 --includes) src/libbddenv.cpp src/bddenv.cpp src/bdd/*.cpp src/instances/*.cpp src/util/*.cpp $(python3-config --ldflags) -L$CPOPT_LIB -lcp -L$CPLEX_LIB -lilocplex -lcplex -L$CONCERT_LIB -lconcert -lm -ldl -pthread -o libbddenvv21o$i.cpython-38-x86_64-linux-gnu.so")

	eval $makelib
	
	sed -i "s/PYBIND11_MODULE(libbddenvv21o$i, m)/PYBIND11_MODULE(libbddenv, m)/g" src/libbddenv.cpp
done
