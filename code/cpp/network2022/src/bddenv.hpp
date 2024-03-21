// --------------------------------------------------
// Multiobjective
// --------------------------------------------------

// General includes
#include <iostream>
#include <cstdlib>

#include "bdd/bdd.hpp"
#include "bdd/bdd_alg.hpp"
#include "bdd/bdd_multiobj.hpp"
#include "util/stats.hpp"
#include "util/util.hpp"
#include "bdd/pareto_frontier.hpp"

// Knapsack includes
#include "instances/knapsack_instance.hpp"
#include "bdd/knapsack_bdd.hpp"

// Set packing / Independent set includes
#include "instances/indepset_instance.hpp"
#include "instances/setpacking_instance.hpp"
#include "bdd/indepset_bdd.hpp"

// // Set covering includes
// #include "instances/setcovering_instance.hpp"
// #include "bdd/setcovering_bdd.hpp"

// // Absolute value instance
// #include "instances/absval_instance.hpp"
// #include "bdd/absval_bdd.hpp"

// // TSP instance
// #include "instances/tsp_instance.hpp"
// #include "mdd/tsp_mdd.hpp"

class BDDEnv
{
public:
    bool reused;

    // ----------------------------------------------------------------
    // Run parameters
    int problem_type;
    bool preprocess;
    int method;
    bool maximization;
    bool dominance;
    int bdd_type;
    int maxwidth;
    vector<int> order;

    // ----------------------------------------------------------------
    // Instance data
    int n_vars;
    int n_objs;
    int n_cons;
    vector<vector<int>> obj_coeffs;
    vector<vector<int>> cons_coeffs;
    vector<int> rhs;

    // ----------------------------------------------------------------
    // Objective coefficients after static/dynamic reordering
    vector<vector<int>> obj_coefficients;

    // ----------------------------------------------------------------
    // BDD topology data

    size_t initial_width, initial_node_count, initial_arcs_count;
    size_t reduced_width, reduced_node_count, reduced_arcs_count;
    vector<size_t> in_degree, max_in_degree_per_layer;
    vector<size_t> initial_num_nodes_per_layer, reduced_num_nodes_per_layer;
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

    BDDEnv();

    ~BDDEnv();

    void reset(int problem_type,
               bool preprocess,
               int method,
               bool maximization,
               bool dominance,
               int bdd_type,
               int maxwidth,
               vector<int> order);

    int set_inst(int n_vars,
                 int n_cons,
                 int n_objs,
                 vector<vector<int>> obj_coeffs,
                 vector<vector<int>> cons_coeffs,
                 vector<int> rhs);

    // int set_inst_tsp(int n_vars,
    //                  int n_objs,
    //                  vector<vector<vector<int>>> obj_coeffs);

    // int set_inst_absval(int n_vars,
    //                     int n_objs,
    //                     double card,
    //                     vector<vector<int>> cons_coeff,
    //                     vector<int> rhs);

    int preprocess_inst();

    int initialize_dd_constructor();

    int generate_dd();

    int generate_next_layer();

    void approximate_layer(int layer,
                           int approx_type = 1,
                           int method = 1,
                           vector<int> states_to_process = {});

    void calculate_bdd_topology_stats(bool is_non_reduced);

    int reduce_dd();

    int compute_pareto_frontier();

    vector<map<string, vector<int>>> get_layer(int);

    vector<vector<map<string, vector<int>>>> get_dd();

    vector<int> get_var_layer();

    map<string, vector<vector<int>>> get_frontier();

    double get_time(int);

    int get_num_nodes_per_layer(int);

private:
    void initialize();

    void clean_memory();

    void restrict_layer(int layer, int method, vector<int> states_to_remove);

    void relax_layer(int layer, int method, vector<int> states_to_merge);

    ParetoFrontier *pareto_frontier;
    // ----------------------------------------------------------------
    // Instances
    KnapsackInstance *inst_kp;
    // SetPackingInstance inst_setpack;
    IndepSetInst *inst_indepset;
    // SetCoveringInstance inst_setcover;
    // AbsValInstance inst_absval;
    // TSPInstance inst_tsp;

    // ----------------------------------------------------------------
    // DD constructor
    KnapsackBDDConstructor kp_bdd_constructor;
    IndepSetBDDConstructor indset_bdd_constructor;
    // SetCoveringBDDConstructor sc_bdd_constructor;
    // AbsValBDDConstructor absval_bdd_constructor;
    // MDDTSPConstructor tsp_mdd_constructor;

    // ----------------------------------------------------------------
    // DD
    BDD *bdd;
    // MDD *mdd;
};