// ----------------------------------------------------------
// BDD Multiobjective Algorithms
// ----------------------------------------------------------

#ifndef BDD_MULTIOBJ_HPP_
#define BDD_MULTIOBJ_HPP_

// #include "../mdd/mdd.hpp"
// #include "../util/util.hpp"
#include "bdd.hpp"
#include "pareto_frontier.hpp"

//
// Multiobjective stats
//
struct MultiObjectiveStats
{
    // Time spent in pareto dominance filtering
    clock_t pareto_dominance_time;
    // Solutions filtered by pareto dominance
    int pareto_dominance_filtered;
    // Layer where coupling happened
    int layer_coupling;

    // Constructor
    MultiObjectiveStats()
        : pareto_dominance_time(0), pareto_dominance_filtered(0), layer_coupling(0)
    {
    }
};

//
// BDD Multiobjective Algorithms
//
struct BDDMultiObj
{
    // Find pareto frontier from top-down approach
    static ParetoFrontier *pareto_frontier_topdown(BDD *bdd, bool maximization = true, const int problem_type = -1, const int dominance_strategy = 0, MultiObjectiveStats *stats = NULL);

    // Find pareto frontier using dynamic layer cutset
    static ParetoFrontier *pareto_frontier_dynamic_layer_cutset(BDD *bdd, bool maximization = true, const int problem_type = -1, const int dominance_strategy = 0, MultiObjectiveStats *stats = NULL);
};

#endif
