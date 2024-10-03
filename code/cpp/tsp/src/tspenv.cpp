// --------------------------------------------------
// Multiobjective
// --------------------------------------------------

// General includes
#include <iostream>

#include "tspenv.hpp"
#include "dd/alg.hpp"

using namespace std;


// //
// // Main function
// //
// int main(int argc, char *argv[])
// {
//     if (argc < 8)
//     {
//         cout << '\n';
//         cout << "Usage: multiobj [input file] [problem type] [preprocess?] [method] [appr-S and T] [dominance]\n";

//         cout << "\n\twhere:";

//         cout << "\n";
//         cout << "\t\tproblem_type = 1: knapsack\n";
//         cout << "\t\tproblem_type = 2: set packing\n";
//         cout << "\t\tproblem_type = 3: set covering\n";
//         cout << "\t\tproblem_type = 4: portfolio optimization\n";
//         cout << "\t\tproblem_type = 5: absolute value\n";
//         cout << "\t\tproblem_type = 6: TSP\n";

//         cout << "\n";
//         cout << "\t\tpreprocess = 0: do not preprocess instance\n";
//         cout << "\t\tpreprocess = 1: preprocess input to minimize BDD size\n";

//         cout << "\n";
//         cout << "\t\tmethod = 1: top-down BFS\n";
//         cout << "\t\tmethod = 2: bottom-up BFS\n";
//         cout << "\t\tmethod = 3: dynamic layer cutset\n";

//         cout << "\n";
//         cout << "\t\tapprox = n m: approximate n-sized S set and m-sized T set (n=0 if disabled)\n";

//         cout << "\n";
//         cout << "\t\tdominance = 0:  disable state dominance\n";
//         cout << "\t\tdominance = 1:  state dominance strategy 1\n";

//         cout << endl;
//         exit(1);
//     }

//     // Read input
//     int problem_type = atoi(argv[2]);
//     bool preprocess = (argv[3][0] == '1');
//     int method = atoi(argv[4]);
//     bool maximization = true;
//     int approx_S = atoi(argv[5]);
//     int approx_T = atoi(argv[6]);
//     int dominance = atoi(argv[7]);

//     // For statistical analysis
//     Stats timers;
//     int bdd_compilation_time = timers.register_name("BDD compilation time");
//     int pareto_time = timers.register_name("BDD pareto time");
//     int approx_time = timers.register_name("BDD approximation time");
//     long int original_width;
//     long int reduced_width;
//     long int original_num_nodes;
//     long int reduced_num_nodes;

//     // Read problem instance and construct BDD
//     vector<vector<int>> obj_coeffs;
//     timers.start_timer(bdd_compilation_time);

//     // --- TSP ---
//     if (problem_type == 6)
//     {

//         clock_t init_tsp = clock();

//         // Read instance
//         TSPInstance inst;
//         inst.read(argv[1]);

//         // Construct MDD
//         clock_t compilation_tsp = clock();

//         MDDTSPConstructor mddCons(&inst);
//         MDD *mdd = mddCons.generate_exact();
//         assert(mdd != NULL);

//         compilation_tsp = clock() - compilation_tsp;

//         // Generate frontier
//         clock_t frontier_tsp = clock();

//         // cout << "\nGenerating frontier..." << endl;
//         MultiObjectiveStats *statsMultiObj = new MultiObjectiveStats;
//         ParetoFrontier *pareto_frontier = NULL;
//         if (method == 1){
//             pareto_frontier = DDParetoAlgorithm::pareto_frontier_topdown(mdd, statsMultiObj);
//         }
//         else if(method ==3){
//             pareto_frontier = DDParetoAlgorithm::pareto_frontier_dynamic_layer_cutset(mdd, statsMultiObj);
//         }
        

//         assert(pareto_frontier != NULL);

//         frontier_tsp = clock() - frontier_tsp;

//         cout << pareto_frontier->get_num_sols() << endl;
//         cout << (double)(compilation_tsp + frontier_tsp) / CLOCKS_PER_SEC << endl;
//         cout << (double)compilation_tsp / CLOCKS_PER_SEC;
//         cout << "\t" << frontier_tsp / CLOCKS_PER_SEC;
//         cout << endl;

//         return 0;
//     }
//     else
//     {
//         cout << "Error - problem type not recognized" << endl;
//         exit(1);
//     }

//     return 0;
// }

vector<int> dynamicBitsetToVector(const boost::dynamic_bitset<> &bitset)
{
    vector<int> result;
    for (boost::dynamic_bitset<>::size_type i = 0; i < bitset.size(); ++i)
    {
        result.push_back(bitset[i]? 1: 0);

    }

    return result;
}

TSPEnv::TSPEnv(){
    initialize();
};

TSPEnv::~TSPEnv(){
    clean_memory();
};


void TSPEnv::initialize(){
    mdd = NULL;
    pareto_frontier = NULL;
};

void TSPEnv::clean_memory(){
    if(mdd != NULL){
        delete mdd;
        mdd = NULL;
    }
    if(pareto_frontier != NULL){
        delete pareto_frontier;
        pareto_frontier = NULL;
    }
};

void TSPEnv::reset(int _method){
    method = _method;
    clean_memory();
}

int TSPEnv::set_inst(int n_cities, int n_objs, vector<vector<vector<int>>> objs){
    inst_tsp = TSPInstance(n_cities, n_objs, objs);

    return 0;
};

int TSPEnv::initialize_dd_constructor(){
    tsp_mdd_constructor = MDDTSPConstructor(&inst_tsp);

    return 0;
}

int TSPEnv::generate_dd(){
    mdd = tsp_mdd_constructor.generate_exact();
    assert(mdd != NULL);

    return 0;
}

int TSPEnv::compute_pareto_frontier(){
    // Generate frontier
    clock_t frontier_tsp = clock();

    // cout << "\nGenerating frontier..." << endl;
    MultiObjectiveStats *statsMultiObj = new MultiObjectiveStats;
    if (method == 1){
        pareto_frontier = DDParetoAlgorithm::pareto_frontier_topdown(mdd, statsMultiObj);
    }
    else if(method ==3){
        pareto_frontier = DDParetoAlgorithm::pareto_frontier_dynamic_layer_cutset(mdd, statsMultiObj);
    }
    assert(pareto_frontier != NULL);

    // frontier_tsp = clock() - frontier_tsp;

    return 0;
}

vector<vector<int>> TSPEnv::get_layer(int l){
    vector<vector<int>> layer;
    layer.reserve(mdd->layers[l].size());

    for (vector<MDDNode *>::iterator it = mdd->layers[l].begin();
            it != mdd->layers[l].end(); ++it)
    {
        vector<int> state = dynamicBitsetToVector((*it)->S);
        state.push_back((*it)->last_city);
        layer.push_back(state);
    }
    
    return layer;
};

vector<vector<vector<int>>> TSPEnv::get_dd(){
    vector<vector<vector<int>>> dd;
    dd.resize(mdd->num_layers - 2);

    for (int l = 1; l < mdd->num_layers - 1; ++l)
    {

        dd[l - 1] = get_layer(l);
    }

    return dd;
};


map<string, vector<vector<int>>> TSPEnv::get_frontier(){
    if (pareto_frontier != NULL)
    {
        // cout << pareto_frontier->sols.size() << endl;
        return pareto_frontier->get_frontier();
    }
    return {};
}