#include <algorithm>
#include <iostream>
#include <fstream>
#include <limits>
#include <vector>
#include <list>

#include "../util/util.hpp"

using namespace std;

//
// Solution representation
//
struct Solution
{
    vector<int> x; // solution
    ObjType *obj;  // objectives

    // Constructors
    Solution() {}
    Solution(vector<int> &_x, ObjType *_obj)
        : x(_x), obj(_obj) {}

    // Print solution
    void print(ostream &out)
    {
        // out << "(";
        for (size_t i = 0; i < x.size(); ++i)
        {
            // if (i > 0)
            // out << ", ";
            out << x[i] << " ";
        }
        // out << ") --> ";
        print_objective(out);
    }

    // Print only objective
    void print_objective(ostream &out)
    {
        for (size_t i = 0; i < NOBJS; ++i)
        {
            if (i > 0)
                out << " ";
            out << obj[i];
        }
    }
};

//
// Solution list
//
typedef list<Solution> SolutionList;
