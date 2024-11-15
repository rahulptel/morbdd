#pragma once

#include "mdd.hpp"
#include "frontier.hpp"

//
// DD Multiobjective Algorithms
//
struct DDParetoAlgorithm
{

    // Find pareto frontier from top-down approach - MDD version
    static ParetoFrontier *pareto_frontier_topdown(MDD *bdd, MultiObjectiveStats *stats);

    // Find pareto frontier using dynamic layer cutset
    static ParetoFrontier *pareto_frontier_dynamic_layer_cutset(MDD *mdd, MultiObjectiveStats *stats);
};
