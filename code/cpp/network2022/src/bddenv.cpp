#include "bddenv.hpp"

BDDEnv::BDDEnv()
{
    reused = false;
    initialize();
    reused = true;
}

BDDEnv::~BDDEnv()
{
    clean_memory();
}

void BDDEnv::clean_memory()
{
    if (inst_kp != NULL)
    {
        delete inst_kp;
    }
    if (inst_indepset != NULL)
    {
        delete inst_indepset;
    }
    if (bdd != NULL)
    {
        delete bdd;
    }
    // if (mdd != NULL)
    // {
    //     delete mdd;
    // }
    if (pareto_frontier != NULL)
    {
        delete pareto_frontier;
    }
}

void BDDEnv::initialize()
{
    maximization = true;

    if (!reused)
    {
        inst_kp = NULL;
        inst_indepset = NULL;
        bdd = NULL;
        // mdd = NULL;
        pareto_frontier = NULL;
    }
    else
    {
        clean_memory();
    }

    initial_width = 0, initial_node_count = 0, initial_arcs_count = 0;
    reduced_width = 0, reduced_node_count = 0, reduced_arcs_count = 0;
    in_degree.clear();
    max_in_degree_per_layer.clear();
    initial_num_nodes_per_layer.clear();
    reduced_num_nodes_per_layer.clear();

    timers.reset_timer(compilation_time);
    timers.reset_timer(pareto_time);
    timers.reset_timer(approx_time);

    nnds = 0;
    n_comparisons = 0;
    num_pareto_sol_per_layer.clear();
    z_sol.clear();
}

void BDDEnv::reset(int _problem_type,
                   bool _preprocess,
                   int _method,
                   bool _maximization,
                   bool _dominance,
                   int _bdd_type = 0,
                   int _maxwidth = -1,
                   vector<int> _order = {})
{
    initialize();

    // 1: knapsack
    // 2: set packing
    // 3: set covering
    // 4: absolute value
    // 5: TSP
    problem_type = _problem_type;

    // 0: do not preprocess instance
    // 1: preprocess input to minimize BDD size
    preprocess = _preprocess;

    // 1: top-down BFS
    // 2: bottom-up BFS
    // 3: dynamic layer cutset
    method = _method;

    // true: solve maximization problem
    // false: solve minimization problem by inverting the signs of the objective
    maximization = _maximization;

    // 0: disable state dominance
    // 1: state dominance strategy 1
    dominance = _dominance;

    // 0: Exact DD
    // 1: Restricted DD
    bdd_type = _bdd_type;

    // Max width of the restricted DD
    maxwidth = _maxwidth;

    // Variable ordering used to construct the DD
    // If no order is provided,
    //      Default variable ordering will be used for knapsack, abs value and tsp
    //      Min-state for set packing
    //      Bandwidth minimization for set covering
    order = _order;
}

int BDDEnv::set_inst(int n_vars,
                     int n_cons,
                     int n_objs,
                     vector<vector<int>> obj_coeffs,
                     vector<vector<int>> cons_coeffs,
                     vector<int> rhs)
{
    // Knapsack problem
    if (problem_type == 1)
    {
        cout << "Setting knapsack problem..." << endl;
        inst_kp = new KnapsackInstance(n_vars, n_cons, n_objs, obj_coeffs, cons_coeffs, rhs);
        // inst_kp->print();

        return 0;
    }
    // Set packing problem
    else if (problem_type == 2)
    {
        cout << "Setting set packing problem..." << endl;
        // _cons_coeff will have variable per constraint array of the shape n_cons x n_vars in constraint
        // variable should be indexed starting from 0
        inst_setpack = SetPackingInstance(n_vars, n_cons, n_objs, obj_coeffs, cons_coeffs);

        // create associated independent set instance
        inst_indepset = inst_setpack.create_indepset_instance();
        // inst_indepset = new IndepSetInst(n_vars, cons_coeffs, obj_coeffs);

        return 0;
    }
    // // Set covering problem
    // else if (problem_type == 3)
    // {
    //     cout << "Setting set covering problem..." << endl;

    //     maximization = false;
    //     // obj_coeffs should be negative

    //     // _cons_coeff will have variable per constraint array of the shape n_cons x n_vars in constraint
    //     // variable should be indexed starting from 0
    //     inst_setcover = SetCoveringInstance(n_vars, n_cons, n_objs, obj_coeffs, cons_coeffs);

    //     return 0;
    // }
    else
    {
        cout << "Invalid problem type! Should be between 1 to 4." << endl;
        return 1;
    }
}

// int BDDEnv::set_inst_absval(int n_vars,
//                             int n_objs,
//                             double card,
//                             vector<vector<int>> cons_coeff,
//                             vector<int> rhs)
// {
//     // Absolute value problem
//     if (problem_type != 4)
//     {
//         cout << "Invalid problem type!" << endl;
//         return 1;
//     }

//     cout << "Setting abs value problem..." << endl;
//     inst_absval = AbsValInstance(n_vars, n_objs, card, cons_coeff, rhs);

//     return 0;
// }

// int BDDEnv::set_inst_tsp(int n_vars,
//                          int n_objs,
//                          vector<vector<vector<int>>> obj_coeffs)
// {
//     if (problem_type != 5)
//     {
//         cout << "Invalid problem type!" << endl;
//         return 1;
//     }
//     // Traveling salesman problem
//     inst_tsp = TSPInstance(n_vars, n_objs, obj_coeffs);
//     return 0;
// }

int BDDEnv::preprocess_inst()
{

    timers.start_timer(compilation_time);

    // Knapsack problem
    if (problem_type == 1)
    {
        if (order.size())
        {
            inst_kp->reorder_coefficients();
        }
    }
    else if (problem_type == 2)
    {
    }
    // // Set covering problem
    // else if (problem_type == 3)
    // {
    //     // preprocess
    //     if (preprocess)
    //     {
    //         inst_setcover.minimize_bandwidth();
    //     }
    // }
    else
    {
        cout << "Invalid problem type!" << endl;
        return 1;
    }
    timers.end_timer(compilation_time);

    return 0;
}

int BDDEnv::initialize_dd_constructor()
{
    timers.start_timer(compilation_time);

    // Knapsack problem
    if (problem_type == 1)
    {
        kp_bdd_constructor = KnapsackBDDConstructor(inst_kp);
        bdd = kp_bdd_constructor.bdd;
    }
    // Set packing problem
    else if (problem_type == 2)
    {

        // generate independent set BDD
        indset_bdd_constructor = IndepSetBDDConstructor(inst_indepset, inst_indepset->obj_coeffs);

        if (order.size())
        {
            indset_bdd_constructor.order_provided = true;
        }
        indset_bdd_constructor.var_layer.clear();
        bdd = indset_bdd_constructor.bdd;
    }
    // // Set covering problem
    // else if (problem_type == 3)
    // {
    //     // create BDD
    //     SetCoveringBDDConstructor sc_bdd_constructor(&inst_setcover, inst_setcover.objs);
    // }
    // // Absolute value problem
    // else if (problem_type == 4)
    // {
    //     AbsValBDDConstructor absval_bdd_constructor(&inst_absval);
    // }
    // // Traveling salesman problem
    // else if (problem_type == 5)
    // {
    //     // Construct MDD
    //     MDDTSPConstructor tsp_mdd_constructor(&inst_tsp);
    // }
    else
    {
        cout << "Invalid problem type!" << endl;
        return 1;
    }

    timers.end_timer(compilation_time);

    return 0;
}

void BDDEnv::calculate_bdd_topology_stats(bool is_non_reduced)
{
    if (is_non_reduced)
    {
        initial_width = bdd->get_width();
        initial_node_count = bdd->get_num_nodes();
        initial_arcs_count = bdd->get_arcs_count();
        initial_num_nodes_per_layer = bdd->get_num_nodes_per_layer();
    }
    else
    {
        reduced_width = bdd->get_width();
        reduced_node_count = bdd->get_num_nodes();
        reduced_arcs_count = bdd->get_arcs_count();
        reduced_num_nodes_per_layer = bdd->get_num_nodes_per_layer();

        in_degree = bdd->get_in_degree();
        max_in_degree_per_layer = bdd->get_max_in_degree_per_layer();
    }
}

int BDDEnv::generate_dd()
{
    timers.start_timer(compilation_time);

    // Knapsack problem
    if (problem_type == 1)
    {
        cout << "Generating knapsack DD..." << endl;
        kp_bdd_constructor.generate_exact();
    }
    // Set packing problem
    else if (problem_type == 2)
    {
        cout << "Generating set packing DD..." << endl;

        indset_bdd_constructor.generate();
        indset_bdd_constructor.alloc.clear_states();
    }
    // // Set covering problem
    // else if (problem_type == 3)
    // {
    //     cout << "Generating set covering DD..." << endl;

    //     // create BDD
    //     bdd = sc_bdd_constructor.generate_exact();
    // }
    // // Absolute value problem
    // else if (problem_type == 4)
    // {
    //     cout << "Generating abs value DD..." << endl;
    //     bdd = absval_bdd_constructor.generate_exact();
    // }
    // // Traveling salesman problem
    // else if (problem_type == 5)
    // {
    //     // Construct MDD
    //     MDD *mdd = tsp_mdd_constructor.generate_exact();
    // }
    else
    {
        cout << "Invalid problem type!" << endl;
        return 1;
    }

    timers.end_timer(compilation_time);

    // if ((problem_type >= 1 && problem_type <= 4 && bdd == NULL) || (problem_type == 5 && mdd == NULL))
    // {
    //     cout << "BDD not built successfully!" << endl;
    //     return 1;
    // }

    // calculate_bdd_topology_stats(true);
    return 0;
}

int BDDEnv::generate_next_layer()
{
    timers.start_timer(compilation_time);
    bool is_done;
    // Knapsack problem
    if (problem_type == 1)
    {
        is_done = kp_bdd_constructor.generate_next_layer();
    }
    // Set packing problem
    else if (problem_type == 2)
    {
        is_done = indset_bdd_constructor.generate_next_layer();
    }
    // // Set covering problem
    // else if (problem_type == 3)
    // {
    //     // create BDD
    //     is_last = sc_bdd_constructor.generate_next_layer();
    // }
    // // Absolute value problem
    // else if (problem_type == 4)
    // {
    //     is_last = absval_bdd_constructor.generate_next_layer();
    // }
    // // Traveling salesman problem
    // else if (problem_type == 5)
    // {
    //     // Construct MDD
    //     is_last = tsp_mdd_constructor.generate_next_layer();
    // }
    else
    {
        cout << "Invalid problem type!" << endl;
        return -1;
    }

    timers.end_timer(compilation_time);

    if (is_done)
    {
        calculate_bdd_topology_stats(true);
    }

    return 0;
}

void BDDEnv::approximate_layer(int layer, int approx_type, int method, vector<int> states_to_process)
{
    if (approx_type == 1)
    {
        restrict_layer(layer, method, states_to_process);
    }
    else if (approx_type == 2)
    {
        relax_layer(layer, method, states_to_process);
    }
}

void BDDEnv::restrict_layer(int layer, int method, vector<int> states_to_remove)
{
    vector<int>::iterator it1;
    vector<Node *> restricted_layer;
    restricted_layer.reserve(bdd->layers[layer].size() - states_to_remove.size());

    if (method == 1 && states_to_remove.size())
    {
        for (int i = 0; i < bdd->layers[layer].size(); ++i)
        {
            it1 = find(states_to_remove.begin(),
                       states_to_remove.end(),
                       i);
            if (it1 != states_to_remove.end())
            {
                bdd->remove_node_ref_prev(bdd->layers[layer][i]);
                states_to_remove.erase(it1);
            }
            else
            {
                restricted_layer.push_back(bdd->layers[layer][i]);
            }
        }
    }
    bdd->layers[layer] = restricted_layer;
}

void BDDEnv::relax_layer(int layer, int method, vector<int> states) {}

int BDDEnv::reduce_dd()
{
    // Reduction time is included in the compilation time
    timers.start_timer(compilation_time);

    // Knapsack problem
    if (problem_type == 1 && bdd == NULL)
    {
        return 1;
    }
    else if (problem_type == 1 && bdd != NULL)
    {
        // Reduction only performed for knapsack problem
        BDDAlg::reduce(bdd);
    }

    timers.end_timer(compilation_time);

    if (problem_type >= 1 && problem_type <= 4)
    {
        calculate_bdd_topology_stats(false);
    }
    else
    {
        // Calculate mdd stats
    }

    return 0;
}

vector<int> dynamicBitsetToVector(const boost::dynamic_bitset<> &bitset)
{
    vector<int> result;

    for (boost::dynamic_bitset<>::size_type i = 0; i < bitset.size(); ++i)
    {
        if (bitset[i])
        {
            result.push_back(i);
        }
    }

    return result;
}

vector<map<string, vector<int>>> BDDEnv::get_layer(int l)
{
    vector<map<string, vector<int>>> layer;
    vector<int> zp, op;
    layer.reserve(bdd->layers[l].size());

    if (problem_type == 1)
    {
        for (vector<Node *>::iterator it = bdd->layers[l].begin();
             it != bdd->layers[l].end(); ++it)
        {
            zp.clear();
            op.clear();

            // Indices of zero prev
            for (vector<Node *>::iterator it1 = (*it)->prev[0].begin(); it1 != (*it)->prev[0].end(); ++it1)
            {
                zp.push_back((*it1)->index);
            }

            // Indices of one prev

            for (vector<Node *>::iterator it1 = (*it)->prev[1].begin(); it1 != (*it)->prev[1].end(); ++it1)
            {
                op.push_back((*it1)->index);
            }

            layer.push_back({{"s", (*it)->weight},
                             {"op", op},
                             {"zp", zp}});
        }
    }
    else if (problem_type == 2)
    {
        for (vector<Node *>::iterator it = bdd->layers[l].begin();
             it != bdd->layers[l].end(); ++it)
        {
            zp.clear();
            op.clear();

            // Indices of zero prev
            for (vector<Node *>::iterator it1 = (*it)->prev[0].begin(); it1 != (*it)->prev[0].end(); ++it1)
            {
                zp.push_back((*it1)->index);
            }

            // Indices of one prev

            for (vector<Node *>::iterator it1 = (*it)->prev[1].begin(); it1 != (*it)->prev[1].end(); ++it1)
            {
                op.push_back((*it1)->index);
            }

            layer.push_back({{"s", dynamicBitsetToVector((*it)->setpack_state)},
                             {"op", op},
                             {"zp", zp}});
        }
    }

    return layer;
}

vector<vector<map<string, vector<int>>>> BDDEnv::get_dd()
{
    vector<vector<map<string, vector<int>>>> dd;
    dd.resize(bdd->num_layers - 2);

    for (int l = 1; l < bdd->num_layers - 1; ++l)
    {

        dd[l - 1] = get_layer(l);
    }

    return dd;
}

vector<int> BDDEnv::get_var_layer()
{
    if (problem_type == 2)
    {
        return indset_bdd_constructor.var_layer;
    }
}

map<string, vector<vector<int>>> BDDEnv::get_frontier()
{
    if (pareto_frontier != NULL)
    {
        cout << pareto_frontier->sols.size() << endl;
        return pareto_frontier->get_frontier();
    }
    return {};
}

double BDDEnv::get_time(int time_type)
{
    if (time_type == 1)
    {
        return timers.get_time(compilation_time);
    }
    else if (time_type == 2)
    {
        return timers.get_time(pareto_time);
    }
    else
    {
        return -1;
    }
}

int BDDEnv::get_num_nodes_per_layer(int layer)
{
    if (bdd != NULL)
    {
        return bdd->layers[layer].size();
    }
    return -1;
}

int BDDEnv::compute_pareto_frontier()
{
    MultiObjectiveStats *statsMultiObj = new MultiObjectiveStats;
    if (problem_type != 5)
    {
        if (bdd == NULL)
        {
            cout << "BDD not constructed! Cannot compute pareto frontier. " << endl;
            return 1;
        }

        // Compute pareto frontier based on methodology
        // cout << "\n\nComputing pareto frontier..." << endl;
        pareto_frontier = NULL;
        timers.start_timer(pareto_time);
        // cout << method << endl;
        if (method == 1)
        {
            // -- Optimal BFS algorithm: top-down --
            pareto_frontier = BDDMultiObj::pareto_frontier_topdown(bdd, maximization, problem_type, dominance, statsMultiObj);
        }
        // else if (method == 2)
        // {
        //     // -- Optimal BFS algorithm: bottom-up --
        //     pareto_frontier = BDDMultiObj::pareto_frontier_bottomup(bdd, maximization, problem_type, dominance, statsMultiObj);
        // }
        // else if (method == 3)
        // {
        //     // -- Dynamic layer cutset --
        //     pareto_frontier = BDDMultiObj::pareto_frontier_dynamic_layer_cutset(bdd, maximization, problem_type, dominance, statsMultiObj);
        // }

        if (pareto_frontier == NULL)
        {
            cout << "\nError - pareto frontier not computed" << endl;
            return 1;
        }
        timers.end_timer(pareto_time);

        return 0;
    }
    // else if (problem_type == 5)
    // {
    //     if (mdd == NULL)
    //     {
    //         cout << "MDD not constructed! Cannot compute pareto frontier. " << endl;
    //         return 1;
    //     }

    //     timers.start_timer(pareto_time);
    //     pareto_frontier = BDDMultiObj::pareto_frontier_dynamic_layer_cutset(mdd, statsMultiObj);
    //     timers.end_timer(pareto_time);

    //     return 0;
    // }
    else
    {
        cout << "Invalid problem name! Cannot compute pareto frontier." << endl;
        return 1;
    }
}