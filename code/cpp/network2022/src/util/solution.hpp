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
    vector<int> x;   // solution
    vector<int> obj; // objectives

    // Constructors
    Solution() {}
    Solution(vector<int> &_x, vector<int> &_obj)
        : x(_x), obj(_obj) {}

    map<string, vector<int>> get()
    {
        map<string, vector<int>> sol;
        sol.insert({"x", x});
        sol.insert({"z", obj});
        return sol;
    }

    void print_obj()
    {
        cout << "(";
        for (int i = 0; i < NOBJS; ++i)
        {
            cout << obj[i];
            if (i < NOBJS - 1)
            {
                cout << ", ";
            }
        }
        cout << ")" << endl;
    }

    void print_x()
    {
        cout << "x: ";
        for (vector<int>::iterator i = x.begin(); i != x.end(); ++i)
        {
            cout << (*i) << " ";
        }
    }
};

//
// Solution list
//
typedef list<Solution> SolutionList;