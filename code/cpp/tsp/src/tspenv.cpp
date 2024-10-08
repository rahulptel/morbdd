// --------------------------------------------------
// Multiobjective
// --------------------------------------------------

// General includes
#include <iostream>

#include "tspenv.hpp"
#include "dd/alg.hpp"

using namespace std;



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

void TSPEnv::reset(){
    clean_memory();
}

int TSPEnv::set_inst(int n_cities, int n_objs, vector<vector<vector<int>>> objs){
    inst_tsp = TSPInstance(n_cities, n_objs, objs);

    return 0;
};

int TSPEnv::initialize_dd_constructor(){
    tsp_mdd_constructor = MDDTSPConstructor(&inst_tsp);
    mdd = tsp_mdd_constructor.mdd;
    return 0;
}

bool TSPEnv::generate_next_layer(){
    return tsp_mdd_constructor.generate_next_layer();
}

int TSPEnv::restrict_layer(int layer, vector<int> states_to_remove){    
    cout << mdd->layers[layer].size() << endl;
    if (states_to_remove.size() >= mdd->layers[layer].size())
    {
        return -1;
    }
    if (states_to_remove.size())
    {
        vector<int>::iterator it1;

        for (int i = 0; i < mdd->layers[layer].size(); ++i)
        {
            it1 = find(states_to_remove.begin(),
                        states_to_remove.end(),
                        i);
            // Remove state refs
            if (it1 != states_to_remove.end())
            {
                mdd->remove_node_refs(mdd->layers[layer][i]);
                states_to_remove.erase(it1);
            }
            
        }

        int i = 0;
        while (i < mdd->layers[layer].size())
        {
            MDDNode *node = mdd->layers[layer][i];
            if (node->in_arcs_list.empty())
            {
                // Remove node from layer
                mdd->layers[layer][i] = mdd->layers[layer].back();
                mdd->layers[layer].pop_back();
                // Remove node
                delete node;
            }
            else
            {
                ++i;
            }
        }
    
        mdd->repair_node_indices(layer);
        tsp_mdd_constructor.fix_state_map();
        
        return 0;
    }
    return -1;
}

int TSPEnv::approximate_layer(int layer, int approx_type, vector<int> states_to_process){
    if (approx_type == 1){
        // cout << "Restrict layer " << endl;
        return restrict_layer(layer, states_to_process);
    }
}



int TSPEnv::generate_dd(){
    mdd = tsp_mdd_constructor.generate_exact();
    assert(mdd != NULL);

    return 0;
}

int TSPEnv::compute_pareto_frontier(){
    method=3;
    // Generate frontier
    clock_t frontier_tsp = clock();

    // cout << "\nGenerating frontier..." << endl;
    MultiObjectiveStats *statsMultiObj = new MultiObjectiveStats;
    if (method == 1){
        pareto_frontier = DDParetoAlgorithm::pareto_frontier_topdown(mdd, statsMultiObj);
    }
    else if(method==3){
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