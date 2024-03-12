#include <algorithm>
#include <iostream>
#include <fstream>
#include <limits>
#include <vector>

#include "../util/util.hpp"

using namespace std;

//
// Solution representation
//
struct Solution
{
    vector<int> x;       // solution
    vector<ObjType> obj; // objectives

    // Constructors
    Solution() {}
    Solution(vector<int> &_x, vector<int> &_obj)
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
        for (size_t i = 0; i < obj.size(); ++i)
        {
            if (i > 0)
                out << " ";
            out << obj[i];
        }
    }

    // Return if solution dominates other
    bool dominates(Solution &sol)
    {
        assert(obj.size() == sol.obj.size());
        bool sol_dominates = true;
        for (size_t i = 0; i < obj.size() && sol_dominates; ++i)
        {
            sol_dominates = (obj[i] >= sol.obj[i]);
        }
        return sol_dominates;
    }
};
