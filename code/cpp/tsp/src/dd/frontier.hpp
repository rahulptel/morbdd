#pragma once


#include "../util/solution.hpp"
#include "../util/util.hpp"
#include "../util/stats.hpp"



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
    void add(Solution &new_sol);

    // Merge pareto frontier solutions into existing set
    // void merge(const ParetoFrontier &frontier);

    // Merge pareto frontier solutions with shift
    void merge(ParetoFrontier &frontier, const ObjType *shift, int last_city);

    // Merge pareto frontier solutions with shift
    void merge(ParetoFrontier &frontier, Solution& offset_sol, bool offset_from_bu);

    // Convolute two nodes from this set to this one
    void convolute(ParetoFrontier &fA, ParetoFrontier &fB);

    // // Remove pre-set dominated solutions
    // void remove_dominated()
    // {
    //     remove_empty();
    // }

    // Get number of solutions
    int get_num_sols() const
    {
        return sols.size();
    }

    // Clear pareto frontier
    void clear()
    {
        sols.resize(0);
    }

    // Print elements in set
    void print_frontier();

    // Sort array in decreasing order
    // void sort_decreasing();

    // // Check consistency
    // bool check_consistency();

    // Obtain sum of points
    ObjType get_sum();

    map<string, vector<vector<int>>> get_frontier();
    // Check if solution is dominated by any element of this set
    // bool is_sol_dominated(const ObjType *sol, const ObjType *shift);

private:
    // Auxiliaries
    ObjType aux[NOBJS];
    ObjType auxB[NOBJS];
    // vector<ObjType *> elems;

    // Remove empty elements
    // void remove_empty();
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
