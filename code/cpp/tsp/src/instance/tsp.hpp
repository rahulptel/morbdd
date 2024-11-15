#pragma once

#include <vector>

using namespace std;

//
// TSP Instance
//
struct TSPInstance
{
    // Number of cities
    int n_cities;
    // Number of objective functions
    int n_objs;
    // Objective functions  (indexed by objective/city/city)
    vector<vector<vector<int>>> objs;

    // Empty Constructor
    TSPInstance() {}

    TSPInstance(int _n_cities, int _n_objs, vector<vector<vector<int>>> _objs)
    :n_cities(_n_cities), n_objs(_n_objs), objs(_objs) {}

    // Read instance based on our model
    void read(const char *filename);
};
