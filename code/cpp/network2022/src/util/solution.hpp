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
    list<int> x;        // solution
    ObjType obj[NOBJS]; // objectives

    // Constructors
    Solution() {}
    Solution(list<int> &_x, ObjType *_obj)
        : x(_x)
    {
        copy(_obj, _obj + NOBJS, obj);
    }

    map<string, vector<int>> get()
    {
        map<string, vector<int>> sol;
        vector<int> x_sol, z_sol;
        for (list<int>::iterator i = x.begin(); i != x.end(); ++i)
        {
            x_sol.push_back(*i);
        }

        for (int i = 0; i < NOBJS; ++i)
        {
            z_sol.push_back(obj[i]);
        }

        sol.insert({"x", x_sol});
        sol.insert({"z", z_sol});
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
        for (list<int>::iterator i = x.begin(); i != x.end(); ++i)
        {
            cout << *i << " ";
        }
    }
};

//
// Solution list
//
typedef list<Solution> SolutionList;
