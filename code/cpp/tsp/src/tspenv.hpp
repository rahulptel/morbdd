#pragma once

#include <vector>

#include "instance/tsp.hpp"
#include "dd/mdd.hpp"
#include "dd/frontier.hpp"
#include "util/stats.hpp"

using namespace std;

class TSPEnv
{
public:
    bool reused;
    int method;    

    // ----------------------------------------------------------------
    // Instance data
    int n_cities;
    int n_objs;
    vector<vector<vector<int>>> objs;
    
    // ----------------------------------------------------------------
    // BDD topology data
    size_t width, node_count, arcs_count;    
    vector<size_t> num_nodes_per_layer;
    vector<size_t> num_pareto_sol_per_layer;
    vector<size_t> num_comparisons_per_layer;

    // ----------------------------------------------------------------
    // Pareto frontier data
    size_t nnds = 0;
    size_t n_comparisons = 0;
    vector<vector<int>> z_sol;

    // For statistical analysis
    // ----------------------------------------------------------------
    Stats timers;
    int compilation_time = timers.register_name("compilation time");
    int pareto_time = timers.register_name("pareto time");
    int approx_time = timers.register_name("approximation time");

    TSPEnv();
    ~TSPEnv();

    void reset(int method);

    int set_inst(int n_cities,
                 int n_objs,
                 vector<vector<vector<int>>> objs);

    int initialize_dd_constructor();


    int generate_dd();

    // int generate_next_layer();

    // int approximate_layer(int layer,
    //                       int approx_type = 1,
    //                       int method = 1,
    //                       vector<int> states_to_process = {});

    // void calculate_bdd_topology_stats(bool is_non_reduced);


    int compute_pareto_frontier();

    vector<vector<int>> get_layer(int);

    vector<vector<vector<int>>> get_dd();

    map<string, vector<vector<int>>> get_frontier();

    // double get_time(int);

    // int get_num_nodes_per_layer(int);

    // void restrict(vector<vector<int>> states_to_remove);

private:
    void initialize();
    void clean_memory();
    // int restrict_layer(int layer, int method, vector<int> states_to_remove);
    // int relax_layer(int layer, int method, vector<int> states_to_merge);

    // ----------------------------------------------------------------
    // Instances
    TSPInstance inst_tsp;
    // ----------------------------------------------------------------
    // DD constructor
    MDDTSPConstructor tsp_mdd_constructor;
    // ----------------------------------------------------------------
    // DD
    MDD *mdd;    
    // ----------------------------------------------------------------
    // Frontier
    ParetoFrontier *pareto_frontier;

};