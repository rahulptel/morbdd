#define BOOST_DYNAMIC_BITSET_DONT_USE_FRIENDS


#include "mdd.hpp"


using namespace std;

//
// MDDArc constructor
//
MDDArc::MDDArc(const int _val, MDDNode *_tail, MDDNode *_head)
    : val(_val), tail(_tail), head(_head), length(0)
{
    assert(tail != NULL);
    assert(head != NULL);
}

//
// MDDArc constructor with lengths
//
MDDArc::MDDArc(const int _val, MDDNode *_tail, MDDNode *_head, int _length)
    : val(_val), tail(_tail), head(_head), length(_length)
{
    assert(tail != NULL);
    assert(head != NULL);
}

//
// Set arc length
//
void MDDArc::set_length(const int _length)
{
    length = _length;
}

//
// MDDNode constructor, where outgoing values range in [0,...,maxvalue)
//
MDDNode::MDDNode(const int _layer, const int _index, const int _max_value)
    : layer(_layer), index(_index), max_value(_max_value)
{
    // allocate list of arcs per value
    out_arc_val.resize(max_value, NULL);
}

//
// Add outgoing arc
//
MDDArc *MDDNode::add_out_arc(MDDNode *head, int val, int length)
{
    assert(head != NULL);
    assert(layer == head->layer - 1);
    assert(val >= 0 && val < max_value);
    assert(out_arc_val[val] == NULL);

    MDDArc *a = new MDDArc(val, this, head, length);
    out_arc_val[val] = a;
    out_arcs_list.push_back(a);
    head->in_arcs_list.push_back(a);

    return a;
}

//
// MDDNode destructor
//
MDDNode::~MDDNode()
{
    for (int j = 0; j < out_arcs_list.size(); ++j)
    {
        assert(out_arcs_list[j] != NULL);
        delete out_arcs_list[j];
    }
}

//
// MDD constructor
//
MDD::MDD(const int _num_layers) : num_layers(_num_layers)
{
    layers.resize(num_layers);
}

//
// MDD destructor
//
MDD::~MDD()
{
    for (int l = 0; l < num_layers; ++l)
    {
        for (int i = 0; i < layers[l].size(); ++i)
        {
            delete layers[l][i];
        }
    }
}

//
// Add a node in layer, considering max value of outgoing arcs
//
MDDNode *MDD::add_node(const int layer, const int max_value)
{
    assert(layer >= 0 && (int)layer < layers.size());
    layers[layer].push_back(new MDDNode(layer, layers[layer].size(), max_value));
    ++num_nodes;
    return layers[layer].back();
}

//
// Remove incoming arc (do not check arc tail)
//
void MDDNode::remove_incoming(MDDArc *a)
{
    assert(a != NULL);
    assert(a->head == this);

    for (int j = 0; j < in_arcs_list.size(); ++j)
    {
        if (in_arcs_list[j] == a)
        {
            in_arcs_list[j] = in_arcs_list.back();
            in_arcs_list.pop_back();
            return;
        }
    }
}

//
// Remove outgoing arc (do not check arc head)
//
void MDDNode::remove_outgoing(MDDArc *a)
{
    assert(a != NULL);
    assert(a->tail == this);

    out_arc_val[a->val] = NULL;
    for (int j = 0; j < out_arcs_list.size(); ++j)
    {
        if (out_arcs_list[j] == a)
        {
            out_arcs_list[j] = out_arcs_list.back();
            out_arcs_list.pop_back();
            return;
        }
    }
}

//
// Add an arc in layer
//
MDDArc *MDD::add_arc(MDDNode *tail, MDDNode *head, int val)
{
    assert(tail != NULL);
    assert(head != NULL);
    assert(tail->layer == head->layer - 1);
    assert(val >= 0 && val < tail->max_value);
    assert(tail->out_arc_val[val] == NULL);

    MDDArc *a = new MDDArc(val, tail, head);
    tail->out_arc_val[val] = a;
    tail->out_arcs_list.push_back(a);
    head->in_arcs_list.push_back(a);

    return a;
}

//
// Add an arc in layer considering length
//
MDDArc *MDD::add_arc(MDDNode *tail, MDDNode *head, int val, int length)
{
    assert(tail != NULL);
    assert(head != NULL);
    assert(tail->layer == head->layer - 1);
    assert(val >= 0 && val < tail->max_value);
    assert(tail->out_arc_val[val] == NULL);

    MDDArc *a = new MDDArc(val, tail, head, length);
    tail->out_arc_val[val] = a;
    tail->out_arcs_list.push_back(a);
    head->in_arcs_list.push_back(a);

    return a;
}

//
// Remove arc
//
void MDD::remove(MDDArc *a)
{
    assert(a != NULL);
    a->tail->remove_outgoing(a);
    a->head->remove_incoming(a);
    delete a;
}

//
// Remove node
//
void MDD::remove(MDDNode *node)
{
    remove_node_refs(node);
    // Deallocate node
    delete node;
}

//
// Remove node references
//
void MDD::remove_node_refs(MDDNode *node){
    // Remove outgoing arcs
    for (int a = 0; a < node->out_arcs_list.size(); ++a)
    {
        MDDArc *arc = node->out_arcs_list[a];
        arc->head->remove_incoming(arc);
        delete arc;
    }
    node->out_arcs_list.clear();
    // Remove incoming arcs
    for (int a = 0; a < node->in_arcs_list.size(); ++a)
    {
        MDDArc *arc = node->in_arcs_list[a];
        arc->tail->remove_outgoing(arc);
        delete arc;
    }
    node->in_arcs_list.clear();
}

//
// Update MDD information
//
void MDD::update()
{
    // Update width
    width = 0;
    for (int l = 0; l < layers.size(); ++l)
    {
        width = std::max(width, (int)layers[l].size());
    }
    // Update maximum arc value in any layer
    max_value = -1;
    for (int l = 0; l < layers.size(); ++l)
    {
        for (int i = 0; i < layers[l].size(); ++i)
        {
            max_value = std::max(max_value, layers[l][i]->max_value);
        }
    }
    // Update number of nodes
    update_num_nodes();
}

//
// Redirect incoming arcs from nodeB to node A
//
void MDD::redirect_incoming(MDDNode *nodeA, MDDNode *nodeB)
{
    for (int j = 0; j < nodeB->in_arcs_list.size(); ++j)
    {
        MDDArc *a = nodeB->in_arcs_list[j];
        a->head = nodeA;
        nodeA->in_arcs_list.push_back(a);
    }
    nodeB->in_arcs_list.clear();
}

//
// Redirect existing arc as incoming to a node
//
void MDD::redirect_incoming(MDDNode *node, MDDArc *in_arc)
{
    node->in_arcs_list.push_back(in_arc);
    in_arc->head = node;
}

//
// Get MDD width
//
int MDD::get_width() const
{
    return width;
}

//
// Get max arc value in any layer
//
int MDD::get_max_value() const
{
    return max_value;
}

//
// Print MDD
//
void MDD::print() const
{
    cout << endl;
    cout << "** MDD **" << endl;
    for (int l = 0; l < num_layers; ++l)
    {
        cout << "\tLayer " << l << endl;
        for (vector<MDDNode *>::const_iterator it = layers[l].begin(); it != layers[l].end(); ++it)
        {
            MDDNode *node = *it;
            cout << "\t\tNode " << node->layer << "," << node->index;
            cout << " - S=";
            for(int i=0; i<node->S.size();++i){
                cout << node->S[i];
            }
            cout << " - last_city=" << node->last_city;
            cout << endl;
            for (int j = 0; j < node->out_arcs_list.size(); ++j)
            {
                MDDArc *a = node->out_arcs_list[j];
                cout << "\t\t\tArc val=" << a->val;
                cout << " - head=" << a->head->index;
                cout << " - tail=" << a->tail->index;
                cout << " - weights={ ";
                for (int o = 0; o < NOBJS; ++o)
                {
                    cout << a->weights[o] << " ";
                }
                cout << "}";
                cout << endl;
            }
        }
    }
    cout << "** Done **" << endl
         << endl;
}

//
// Get root node
//
MDDNode *MDD::get_root() const
{
    assert(num_layers > 0);
    assert(layers[0].size() > 0);

    return layers[0][0];
}

//
// Get terminal node
//
MDDNode *MDD::get_terminal() const
{
    assert(num_layers > 0);
    assert(layers[num_layers - 1].size() > 0);

    return layers[num_layers - 1][0];
}

//
// Remove nodes with zero outgoing arcs
//
void MDD::remove_dangling_outgoing()
{
    for (int l = num_layers - 2; l >= 0; --l)
    {
        int i = 0;
        while (i < layers[l].size())
        {
            MDDNode *node = layers[l][i];
            if (node->out_arcs_list.empty())
            {
                // Remove node from layer
                layers[l][i] = layers[l].back();
                layers[l].pop_back();
                // Remove node
                remove(node);
            }
            else
            {
                ++i;
            }
        }
    }
}



//
// Remove nodes with zero incoming arcs
//
void MDD::remove_dangling_incoming()
{
    for (int l = 1; l < num_layers; ++l)
    {
        int i = 0;
        while (i < layers[l].size())
        {
            MDDNode *node = layers[l][i];
            if (node->in_arcs_list.empty())
            {
                // Remove node from layer
                layers[l][i] = layers[l].back();
                layers[l].pop_back();
                // Remove node
                remove(node);
            }
            else
            {
                ++i;
            }
        }
    }
}

//
// Remove dangling nodes
//
void MDD::remove_dangling()
{
    remove_dangling_incoming();
    remove_dangling_outgoing();
}

//
// Check MDD consistency
//
bool MDD::check_consistency()
{
    // cout << endl << endl;
    for (int l = 0; l < num_layers; ++l)
    {
        // cout << "Layer " << l << endl;
        if (layers[l].size() == 0)
        {
            // cout << "Error: layer " << l << " is empty." << endl;
            return false;
        }
        for (int i = 0; i < layers[l].size(); ++i)
        {
            MDDNode *node = layers[l][i];
            // cout << "\tMDDNode " << node->index << endl;

            if (l < num_layers - 1)
            {
                if (node->out_arcs_list.size() == 0)
                {
                    cout << "Error: node " << node->layer << "," << node->index;
                    cout << " has zero outgoing arcs." << endl;
                    return false;
                }
                for (int j = 0; j < node->out_arcs_list.size(); ++j)
                {
                    MDDArc *a = node->out_arcs_list[j];
                    if (a != node->out_arc_val[a->val])
                    {
                        cout << "Error: ";
                        cout << "node " << node->layer << "," << node->index;
                        cout << " - arc: " << a->head->index << "," << a->val;
                        cout << " does not correspond to correct out_arcs_val" << endl;
                        return false;
                    }
                    const MDDNode *head = a->head;
                    bool found = false;
                    for (int k = 0; k < head->in_arcs_list.size() && !found; ++k)
                    {
                        found = (head->in_arcs_list[k] == a);
                    }
                    if (!found)
                    {
                        cout << "Error: ";
                        cout << "node " << node->layer << "," << node->index;
                        cout << " - arc: " << a->head->index << "," << a->val;
                        cout << " was not found in head incoming list" << endl;
                        return false;
                    }
                }
                for (int v = 0; v < node->max_value; ++v)
                {
                    if (node->out_arc_val[v] == NULL)
                    {
                        continue;
                    }
                    MDDArc *a = node->out_arc_val[v];
                    bool found = false;
                    for (int j = 0; j < node->out_arcs_list.size() && !found; ++j)
                    {
                        found = (a == node->out_arcs_list[j]);
                    }
                    if (!found)
                    {
                        cout << "Error: ";
                        cout << "node " << node->layer << "," << node->index;
                        cout << " - arc: " << a->head->index << "," << a->val;
                        cout << " was not found in outgoing list" << endl;
                        return false;
                    }
                }
            }
            if (l > 1)
            {
                if (node->in_arcs_list.size() == 0)
                {
                    cout << "Error: node " << node->layer << "," << node->index;
                    cout << " has zero incoming arcs." << endl;
                    return false;
                }
                for (int j = 0; j < node->in_arcs_list.size(); ++j)
                {
                    MDDArc *a = node->in_arcs_list[j];
                    bool found = false;
                    for (int k = 0; k < a->tail->out_arcs_list.size() && !found; ++k)
                    {
                        found = (a->tail->out_arcs_list[k] == a);
                    }
                    if (!found)
                    {
                        cout << "Error: ";
                        cout << "node " << node->layer << "," << node->index;
                        cout << " - arc tail: " << a->tail->index << "," << a->val;
                        cout << " was not found in outgoing list of tail" << endl;
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

//
// Repair MDD node indices
//
void MDD::repair_node_indices(int l)
{
    for (int i = 0; i < layers[l].size(); ++i)
    {
        layers[l][i]->index = i;
    }
}

//
// Repair MDD node indices
//
void MDD::repair_node_indices()
{
    for (int l = 0; l < num_layers; ++l)
    {
        repair_node_indices(l);
    }
}

// Update number of nodes
void MDD::update_num_nodes()
{
    num_nodes = 0;
    for (int l = 0; l < num_layers; ++l)
    {
        num_nodes += layers[l].size();
    }
}

//
// Constructor
//
MDDTSPConstructor::MDDTSPConstructor(TSPInstance *_inst)
    : inst(_inst)
{
    // Initialize MDD
    mdd = new MDD(inst->n_cities + 2);

    // array of cities to visit
    for (int i = 0; i < inst->n_cities; ++i)
    {
        exact_vals.push_back(i);
    }

    // Create root node
    MDDNode *root_node = mdd->add_node(0, inst->n_cities);
    root_node->S.resize(exact_vals.size());
    root_node->S.reset();
    root_node->last_city = -1;

    // This is used for branching, but in this case only the first city (0) is fixed.
    fixed_vals.push_back(0);

    is_fixed = new bool[inst->n_cities];
    memset(is_fixed, false, sizeof(bool) * inst->n_cities);
    for (int c = 0; c < fixed_vals.size(); ++c)
    {
        is_fixed[fixed_vals[c]] = true;
    }

    
    // Values that have to be exact
    is_city_exact = new bool[inst->n_cities];
    memset(is_city_exact, false, sizeof(bool) * inst->n_cities);
    for (int c = 0; c < exact_vals.size(); ++c)
    {
        is_city_exact[exact_vals[c]] = true;
    }

    map.resize(inst->n_cities);
    std::fill(map.begin(), map.end(), -1);
    int pos = 0;
    for (int c = 0; c < exact_vals.size(); ++c)
    {
        map[exact_vals[c]] = pos++;
    }

    // Initialize node map    
    iter = 0;
    next = 1;

    // Add root node to state set
    root_node = mdd->layers[0][0];
    states[iter][root_node] = root_node;

    // Initialize node buffer for general operations
    node_buffer = new MDDNode;

    // Create layers associated with fixed variables
    // All fixed arcs have a zero length
    for (int l = 0; l < fixed_vals.size(); ++l)
    {
        assert(mdd->layers[l].size() == 1);

        // Reset next state
        states[next].clear();

        // Obtain node
        MDDNode *node = mdd->layers[l].back();

        // Create next layer node associated with fixed variable
        MDDNode *new_node = mdd->add_node(l + 1, inst->n_cities);
        new_node->S = node->S;
        new_node->S[fixed_vals[l]] = true;
        new_node->last_city = fixed_vals[l];

        states[next][new_node] = new_node;
        MDDArc *arc = node->add_out_arc(new_node, fixed_vals[l], 0);

        arc->weights = new ObjType[inst->n_objs];
        memset(arc->weights, 0, sizeof(ObjType) * inst->n_objs);

        // invert iter and next
        next = !next;
        iter = !iter;
    }

    l = fixed_vals.size();

}

bool MDDTSPConstructor::generate_next_layer(){
    // cout << "Layer " << l << endl;

    if (l < inst->n_cities){
        // Reset next state
        states[next].clear();

        // Extend each state
        BOOST_FOREACH (StateNodeMap::value_type i, states[iter])
        {
            MDDNode *node_state = i.first;
            MDDNode *node = i.second;

            for (int v = 0; v < inst->n_cities; ++v)
            {
                // Check if it is possible to extend state
                if (!node_state->S[v])
                {
                    // Visited city state
                    node_buffer->S = node_state->S;
                    node_buffer->S[v] = true;
                    // Last city
                    node_buffer->last_city = v;

                    // Check if state exists in the next layer
                    it = states[next].find(node_buffer);
                    if (it == states[next].end())
                    {
                        MDDNode *new_node = mdd->add_node(l + 1, inst->n_cities);
                        new_node->S = node_buffer->S;
                        new_node->last_city = node_buffer->last_city;

                        states[next][new_node] = new_node;

                        MDDArc *arc = node->add_out_arc(new_node, v, 0);
                        arc->weights = new ObjType[inst->n_objs];
                        for (int o = 0; o < inst->n_objs; ++o)
                        {
                            arc->weights[o] = (-1) * inst->objs[o][node_state->last_city][v];
                        }
                    }
                    else
                    {                        
                        MDDNode *head = it->second;
                        MDDArc *arc = node->add_out_arc(head, v, 0);
                        arc->weights = new ObjType[inst->n_objs];
                        for (int o = 0; o < inst->n_objs; ++o)
                        {
                            arc->weights[o] = (-1) * inst->objs[o][node_state->last_city][v];
                        }
                    }
                }
            }
            assert(node->out_arcs_list.size() > 0);
        }

        // invert iter and next
        next = !next;
        iter = !iter;
        ++l;

        return false;
    }
    else if(l == inst->n_cities){
        // Create terminal node in last layer
        MDDNode *terminal = mdd->add_node(inst->n_cities + 1, inst->n_cities);
        terminal->last_city = -1;
        for (MDDNode *node : mdd->layers[inst->n_cities])
        {
            MDDArc *arc = node->add_out_arc(terminal, 0, 0);
            arc->weights = new ObjType[inst->n_objs];
            for (int o = 0; o < inst->n_objs; ++o)
            {
                arc->weights[o] = (-1) * inst->objs[o][node->last_city][0];
            }
        }
        ++l;        
        mdd->update();
        assert(mdd->check_consistency());

        return true;
    }

    return true;

}

//
// Generate exact MDD
//
MDD *MDDTSPConstructor::generate_exact()
{
    bool is_done=false;
    do{
        is_done = generate_next_layer();
        cout << l << endl;
    }
    while (!is_done);
    
    return mdd;
}


void MDDTSPConstructor::fix_state_map(){
    // If the last layer is approximated update the states[iter]
	if (states[iter].size() > mdd->layers[l].size())
	{
		states[iter].clear();
		for (int k = 0; k < mdd->layers[l].size(); ++k)
		{
			states[iter][mdd->layers[l][k]] = mdd->layers[l][k];
		}
	}
}