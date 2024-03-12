// ----------------------------------------------------------
// Pareto Frontier classes
// ----------------------------------------------------------

#ifndef PARETO_FRONTIER_HPP_
#define PARETO_FRONTIER_HPP_

#define DOMINATED -9999999
#define EPS 0.0001

#include <algorithm>
#include <iostream>
#include <fstream>
#include <limits>
#include <vector>
#include <list>
#include <iterator>

#include "../util/util.hpp"
#include "../util/solution.hpp"

using namespace std;

//
// Pareto Frontier struct
//
class ParetoFrontier
{
public:
    // (Flat) array of solutions
    // vector<ObjType> sols;
    SolutionList sols;

    // Add element to set
    // void add(ObjType *elem);
    void add(Solution &sol);

    // Merge pareto frontier solutions into existing set
    // void merge(const ParetoFrontier &frontier);

    // Merge pareto frontier solutions with shift
    void merge(ParetoFrontier &frontier, ObjType *shift);

    // Merge pareto frontier solutions with shift
    // void merge_after_convolute(ParetoFrontier &frontier, Solution &sol, bool reverse_outer);

    // void merge(const ParetoFrontier &frontier, const ObjType *shift, int arc_type);

    // Convolute two nodes from this set to this one
    // void convolute(ParetoFrontier &fA, ParetoFrontier &fB);

    // Remove pre-set dominated solutions
    void remove_dominated()
    {
        remove_empty();
    }

    // Get number of solutions
    int get_num_sols()
    {
        return sols.size();
    }

    // Clear pareto frontier
    void clear()
    {
        sols.clear();
    }

    // Print elements in set
    void print();

    // Check consistency
    bool check_consistency();

    // Obtain sum of points
    ObjType get_sum();

    map<string, vector<vector<int>>> get_frontier();

private:
    // Auxiliaries
    // ObjType aux[NOBJS];
    ObjType auxB[NOBJS];
    vector<ObjType *> elems;
    vector<int> aux;

    // Remove empty elements
    void remove_empty();
};

//
// Pareto frontier manager
//
class ParetoFrontierManager
{
public:
    // Constructor
    ParetoFrontierManager() {}
    ParetoFrontierManager(int size)
    {
        frontiers.reserve(size);
    }

    // Destructor
    ~ParetoFrontierManager()
    {
        for (int i = 0; i < frontiers.size(); ++i)
        {
            delete frontiers[i];
        }
    }

    // Request pareto frontier
    ParetoFrontier *request()
    {
        if (frontiers.empty())
        {
            return new ParetoFrontier;
        }
        ParetoFrontier *f = frontiers.back();
        f->clear();
        frontiers.pop_back();
        return f;
    }

    // Return frontier to allocation
    void deallocate(ParetoFrontier *frontier)
    {
        frontiers.push_back(frontier);
    }

    // Preallocated array set
    vector<ParetoFrontier *> frontiers;
};

// Modify
//
// Add element to set
//
inline void ParetoFrontier::add(Solution &sol)
{
    bool dominates;
    bool dominated;
    for (SolutionList::iterator it = sols.begin(); it != sols.end();)
    {
        // check status of foreign solution w.r.t. current frontier solution
        dominates = true;
        dominated = true;
        for (int o = 0; o < NOBJS && (dominates || dominated); ++o)
        {
            dominates &= (sol.obj[o] >= it->obj[o]);
            dominated &= (sol.obj[o] <= it->obj[o]);
        }
        if (dominated)
        {
            // if foreign solution is dominated, nothing needs to be done
            return;
        }
        else if (dominates)
        {
            // solution dominates iterate
            it = sols.erase(it);
        }
        else
        {
            ++it;
        }
    }
    cout << sol.x.size() << endl;
    cout << sol.obj[0] << " " << sol.obj[1] << " " << sol.obj[2] << endl;
    sols.insert(sols.end(), sol);
}

//
// Merge pareto frontier into existing set considering shift
//
inline void ParetoFrontier::merge(ParetoFrontier &frontier, ObjType *shift)
{
    bool must_add;
    bool dominates;
    bool dominated;

    // Get arc type
    bool is_zero_arc = true;
    for (int o = 0; o < NOBJS; ++o)
    {
        is_zero_arc &= shift[o] == 0;
    }

    cout << "Frontier " << frontier.sols.size() << endl;
    cout << "Shift " << shift[0] << " " << shift[1] << " " << shift[2] << endl;

    // add artificial solution to avoid rechecking dominance between elements in the
    // set to be merged
    Solution dummy;
    SolutionList::iterator end = sols.insert(sols.end(), dummy);
    for (SolutionList::iterator itParent = frontier.sols.begin();
         itParent != frontier.sols.end();
         ++itParent)
    {
        Solution parent = *itParent;

        must_add = true;
        // Compare the incoming aux solution with the sols on the current node

        for (SolutionList::iterator itCurr = sols.begin();
             itCurr != end;)
        {
            // check status of foreign solution w.r.t. current frontier solution
            dominates = true;
            dominated = true;
            // cout << "Cmp "
            //      << "(" << aux[0] << ", " << aux[1] << ", " << aux[2] << "), (" << itCurr->obj[0] << ", " << itCurr->obj[1] << ", " << itCurr->obj[2] << ")" << endl;
            Solution curr = *itCurr;

            for (int o = 0; o < NOBJS && (dominates || dominated); ++o)
            {
                dominates &= (parent.obj[o] + shift[o] >= curr.obj[o]);
                dominated &= (parent.obj[o] + shift[o] <= curr.obj[o]);
                // dominates &= (aux[o] >= itCurr->obj[o]);
                // dominated &= (aux[o] <= itCurr->obj[o]);
                // dominates &= (itParent->obj[o] + shift[o] >= itCurr->obj[o]);
                // dominated &= (itParent->obj[o] + shift[o] <= itCurr->obj[o]);
            }
            cout << "Dominates " << dominates << endl;
            cout << "Dominated " << dominated << endl;
            if (dominated)
            {
                // if foreign solution is dominated, just stop loop
                must_add = false;
                break;
            }
            else if (dominates)
            {
                itCurr = sols.erase(itCurr);
            }
            else
            {
                ++itCurr;
            }
        }
        // if solution has not been added already, append element to the end
        if (must_add)
        {
            cout << "Adding element " << sols.size() << endl;

            Solution new_solution(parent.x, parent.obj);
            if (is_zero_arc)
            {
                new_solution.x.push_back(0);
            }
            else
            {
                new_solution.x.push_back(1);
                for (int i = 0; i < NOBJS; ++i)
                {
                    new_solution.obj[i] += shift[i];
                }
            }
            sols.insert(sols.end(), new_solution);
            sols.size();
        }
    }
    sols.erase(end);
    cout << sols.size() << endl;
}

// //
// // Merge pareto frontier into existing set considering shift
// //
// inline void ParetoFrontier::merge_after_convolute(ParetoFrontier &frontier, Solution &sol, bool reverse_outer)
// {
//     bool must_add;
//     bool dominates;
//     bool dominated;

//     // add artificial solution to avoid rechecking dominance between elements in the
//     // set to be merged
//     Solution dummy;
//     SolutionList::iterator end = sols.insert(sols.end(), dummy);

//     for (SolutionList::iterator itParent = frontier.sols.begin();
//          itParent != frontier.sols.end();
//          ++itParent)
//     {
//         // update auxiliary
//         for (int o = 0; o < NOBJS; ++o)
//         {
//             aux[o] = itParent->obj[o] + sol.obj[o];
//         }
//         must_add = true;
//         // Compare the incoming aux solution with the sols on the current node
//         for (SolutionList::iterator itCurr = sols.begin();
//              itCurr != end;)
//         {
//             // check status of foreign solution w.r.t. current frontier solution
//             dominates = true;
//             dominated = true;
//             for (int o = 0; o < NOBJS && (dominates || dominated); ++o)
//             {
//                 dominates &= (aux[o] >= itCurr->obj[o]);
//                 dominated &= (aux[o] <= itCurr->obj[o]);
//             }
//             if (dominated)
//             {
//                 // if foreign solution is dominated, just stop loop
//                 must_add = false;
//                 break;
//             }
//             else if (dominates)
//             {
//                 itCurr = sols.erase(itCurr);
//             }
//             else
//             {
//                 ++itCurr;
//             }
//         }
//         // if solution has not been added already, append element to the end
//         if (must_add)
//         {
//             if (reverse_outer)
//             {
//                 reverse(itParent->x.begin(), itParent->x.end());
//                 sol.x.insert(sol.x.end(), itParent->x.begin(), itParent->x.end());
//                 sols.insert(sols.end(), Solution(sol.x, aux));
//             }
//             else
//             {
//                 reverse(sol.x.begin(), sol.x.end());
//                 itParent->x.insert(itParent->x.end(), sol.x.begin(), sol.x.end());
//                 sols.insert(sols.end(), Solution(itParent->x, aux));
//             }
//         }
//     }
//     sols.erase(end);
// }

//
// Print elements in set
//
inline void ParetoFrontier::print()
{
    for (SolutionList::iterator it = sols.begin(); it != sols.end(); ++it)
    {
        cout << "(";
        for (int o = 0; o < NOBJS - 1; ++o)
        {
            cout << it->obj[o] << ",";
        }
        cout << it->obj[NOBJS - 1] << ")";
        cout << endl;
    }
}

// //
// // Convolute two nodes from this set to this one
// //
// inline void ParetoFrontier::convolute(ParetoFrontier &fA, ParetoFrontier &fB)
// {
//     if (fA.sols.size() < fB.sols.size())
//     {
//         for (it = fA.sols.begin(); it != fA.sols.end(); ++it)
//         {
//             // std::copy(fA.sols.begin() + j, fA.sols.begin() + j + NOBJS, auxB);
//             merge_after_convolute(fB, *it, true);
//         }
//     }
//     else
//     {
//         for (it = fB.sols.begin(); it != fB.sols.end(); ++it)
//         {
//             // std::copy(fB.sols.begin() + j, fB.sols.begin() + j + NOBJS, auxB);
//             merge_after_convolute(fA, *it, false);
//         }
//     }
// }

//
// Obtain sum of points
//
inline ObjType ParetoFrontier::get_sum()
{
    ObjType sum = 0;
    for (SolutionList::iterator it = sols.begin(); it != sols.end(); ++it)
    {
        for (int o = 0; o < NOBJS; ++o)
        {
            sum += it->obj[o];
        }
    }
    return sum;
}

inline map<string, vector<vector<int>>> ParetoFrontier::get_frontier()
{
    vector<vector<int>> x_sols;
    vector<vector<int>> z_sols;
    x_sols.reserve(sols.size());
    z_sols.reserve(sols.size());
    for (SolutionList::iterator it = sols.begin(); it != sols.end(); ++it)
    {
        x_sols.push_back(it->x);

        vector<int> z_vec;
        z_vec.reserve(NOBJS);
        for (int i = 0; i < NOBJS; ++i)
        {
            z_vec[i] = it->obj[i];
        }
        z_sols.push_back(z_vec);
    }

    map<string, vector<vector<int>>> frontier;
    frontier.insert({"x", x_sols});
    frontier.insert({"z", z_sols});

    return frontier;
}

#endif
