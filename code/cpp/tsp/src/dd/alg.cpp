#include "alg.hpp"

//
// Comparator for node selection in convolution
//
struct CompareMDDNode
{
    bool operator()(const MDDNode *nodeA, const MDDNode *nodeB)
    {
        return (nodeA->pareto_frontier->get_sum() + nodeA->pareto_frontier_bu->get_sum()) > (nodeB->pareto_frontier->get_sum() + nodeB->pareto_frontier_bu->get_sum());
    }
};

//
// Topdown value of a node (for dynamic layer selection)
//
int topdown_layer_value(MDD *mdd, MDDNode *node)
{
    // int total = 0;
    // for (MDDArc* arc : node->out_arcs_list) {
    // 	total += node->pareto_frontier->get_num_sols();
    // }
    // return total;
    return node->pareto_frontier->get_num_sols() * node->out_arcs_list.size();
}

//
// Bottomup value of a node (for dynamic layer selection)
//
int bottomup_layer_value(MDD *mdd, MDDNode *node)
{
    // int total = 0;
    // for (MDDArc* arc : node->out_arcs_list) {
    // 	total += node->pareto_frontier_bu->get_num_sols();
    // }
    // return 1.5*total;
    return 1.5 * node->pareto_frontier_bu->get_num_sols() * node->in_arcs_list.size();
}

//
// Expand pareto frontier / topdown version
//
void expand_layer_topdown(MDD *mdd, const int l, ParetoFrontierManager *mgmr)
{
    MDDNode *node = NULL;
    for (int i = 0; i < mdd->layers[l].size(); ++i)
    {
        node = mdd->layers[l][i];
        // Request frontier
        node->pareto_frontier = mgmr->request();

        // add incoming one arcs
        for (MDDArc *arc : node->in_arcs_list)
        {
            node->pareto_frontier->merge(*(arc->tail->pareto_frontier), arc->weights, arc->tail->last_city);
        }
    }

    // deallocate previous layer
    for (int i = 0; i < mdd->layers[l - 1].size(); ++i)
    {
        mgmr->deallocate(mdd->layers[l - 1][i]->pareto_frontier);
        // delete mdd->layers[l - 1][i];
    }
}

//
// Expand pareto frontier / topdown version
//
void expand_layer_bottomup(MDD *mdd, const int l, ParetoFrontierManager *mgmr)
{
    MDDNode *node = NULL;
    for (int i = 0; i < mdd->layers[l].size(); ++i)
    {
        node = mdd->layers[l][i];
        // Request frontier
        node->pareto_frontier_bu = mgmr->request();

        // add incoming one arcs
        for (MDDArc *arc : node->out_arcs_list)
        {
            node->pareto_frontier_bu->merge(*(arc->head->pareto_frontier_bu), arc->weights, arc->tail->last_city);
        }
    }

    // deallocate next layer
    for (int i = 0; i < mdd->layers[l + 1].size(); ++i)
    {
        mgmr->deallocate(mdd->layers[l + 1][i]->pareto_frontier_bu);
        // delete mdd->layers[l + 1][i];
    }
}

//
// Find pareto frontier using top-down approach for MDDs
//
ParetoFrontier *DDParetoAlgorithm::pareto_frontier_topdown(MDD *mdd, MultiObjectiveStats *stats)
{
    // Initialize stats
    stats->pareto_dominance_time = 0;
    stats->pareto_dominance_filtered = 0;
    clock_t time_filter = 0, init;

    // Initialize manager
    ParetoFrontierManager *mgmr = new ParetoFrontierManager(mdd->get_width());

    // Root node
    // ObjType zero_array[NOBJS];
    // memset(zero_array, 0, sizeof(ObjType) * NOBJS);
    vector<int> x;
	vector<int> obj(NOBJS, 0);
	Solution rootSol(x, obj);
	
    mdd->get_root()->pareto_frontier = mgmr->request();
    mdd->get_root()->pareto_frontier->add(rootSol);

    // Generate frontiers for each node
    for (int l = 1; l < mdd->num_layers; ++l)
    {
        cout << "Layer " << l << endl;
        for (MDDNode *node : mdd->layers[l])
        {
            int id = node->index;

            // Request frontier
            node->pareto_frontier = mgmr->request();

            // add incoming one arcs
            for (MDDArc *arc : node->in_arcs_list)
            {
                node->pareto_frontier->merge(*(arc->tail->pareto_frontier), arc->weights, arc->tail->last_city);
            }
        }

        // Deallocate frontier from previous layer
        for (MDDNode *node : mdd->layers[l - 1])
        {
            mgmr->deallocate(node->pareto_frontier);
        }
    }

    // Erase memory
    delete mgmr;
    return mdd->get_terminal()->pareto_frontier;
}

//
// Find pareto frontier using dynamic layer cutset
//
ParetoFrontier *DDParetoAlgorithm::pareto_frontier_dynamic_layer_cutset(MDD *mdd, MultiObjectiveStats *stats)
{
    // Create pareto frontier manager
    ParetoFrontierManager *mgmr = new ParetoFrontierManager(mdd->get_width());

    // Create root and terminal frontiers
    // ObjType sol[NOBJS];
    // memset(sol, 0, sizeof(ObjType) * NOBJS);
    vector<int> x1;
	vector<int> obj1(NOBJS, 0);
	Solution rootSol(x1, obj1);    
    mdd->get_root()->pareto_frontier = mgmr->request();
    mdd->get_root()->pareto_frontier->add(rootSol);

    vector<int> x2;
	vector<int> obj2(NOBJS, 0);
	Solution termSol(x2, obj2);
    mdd->get_terminal()->pareto_frontier_bu = mgmr->request();
    mdd->get_terminal()->pareto_frontier_bu->add(termSol);

    // Initialize stats
    stats->pareto_dominance_time = 0;
    stats->pareto_dominance_filtered = 0;
    clock_t time_filter = 0, init;

    // Current layers
    int layer_topdown = 0;
    int layer_bottomup = mdd->num_layers - 1;

    // Value of layer
    int val_topdown = 0;
    int val_bottomup = 0;

    int old_topdown = -1;
    while (layer_topdown != layer_bottomup)
    {
        cout << "Layer topdown: " << layer_topdown << " - layer bottomup: " << layer_bottomup << endl;
        if (val_topdown <= val_bottomup)
        {
            // Expand topdown
            expand_layer_topdown(mdd, ++layer_topdown, mgmr);
            // Recompute layer value
            val_topdown = 0;
            for (int i = 0; i < mdd->layers[layer_topdown].size(); ++i)
            {
                val_topdown += topdown_layer_value(mdd, mdd->layers[layer_topdown][i]);
            }
        }
        else
        {
            // Expand layer bottomup
            expand_layer_bottomup(mdd, --layer_bottomup, mgmr);
            // Recompute layer value
            val_bottomup = 0;
            for (int i = 0; i < mdd->layers[layer_bottomup].size(); ++i)
            {
                val_bottomup += bottomup_layer_value(mdd, mdd->layers[layer_bottomup][i]);
            }
        }        
    }


    // Save stats
    stats->layer_coupling = layer_topdown;
    cout << "Coupling layer : " << stats->layer_coupling << endl;

    vector<MDDNode *> &cutset = mdd->layers[layer_topdown];
    sort(cutset.begin(), cutset.end(), CompareMDDNode());

    // Compute expected frontier size
    long int expected_size = 0;
    for (int i = 0; i < cutset.size(); ++i)
    {
        expected_size += cutset[i]->pareto_frontier->get_num_sols() * cutset[i]->pareto_frontier_bu->get_num_sols();
    }
    expected_size = 10000;

    ParetoFrontier *paretoFrontier = new ParetoFrontier;
    // paretoFrontier->sols.reserve(expected_size * NOBJS);

    for (int i = 0; i < cutset.size(); ++i)
    {
        MDDNode *node = cutset[i];
        assert(node->pareto_frontier != NULL);
        assert(node->pareto_frontier_bu != NULL);

        paretoFrontier->convolute(*(node->pareto_frontier), *(node->pareto_frontier_bu));
    }

    // deallocate manager
    delete mgmr;

    // return pareto frontier
    return paretoFrontier;
}