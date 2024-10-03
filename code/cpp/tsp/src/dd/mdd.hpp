#pragma once

#include <iostream>
#include <vector>
#include <boost/dynamic_bitset.hpp>
#include <boost/unordered_map.hpp>
#include <boost/foreach.hpp>

#include "frontier.hpp"
#include "../util/util.hpp"
#include "../instance/tsp.hpp"


using namespace std;


// ----------------------------------------------------------
// Decision Diagram Data Structure
// ----------------------------------------------------------
//

struct MDDNode;
struct MDDArc;
struct MDD;

//
// DD MDDNode
//
struct MDDNode
{
    // MDDNode layer
    const int layer;
    // MDDNode index
    int index;
    // Max value of outgoing arcs
    const int max_value;
    // Incoming arcs
    vector<MDDArc *> in_arcs_list;
    // Outgoing arc list
    vector<MDDArc *> out_arcs_list;
    // Outgoing arc per value
    vector<MDDArc *> out_arc_val;
    // Set of cities visited
    boost::dynamic_bitset<> S;
    // Last city visited
    int last_city;
    // Length from root to node
    int length;
    // Pareto frontier
    ParetoFrontier *pareto_frontier;
    // Pareto frontier from the bottom
    ParetoFrontier *pareto_frontier_bu;

    // MDDNode constructor, where outgoing values range in [0, _maxvalue)
    MDDNode(const int _layer, const int _index, int _max_value);

    // Empty node constructor
    MDDNode() : layer(-1), max_value(-1) {}

    // MDDNode destructor
    ~MDDNode();

    // Add outgoing arc
    MDDArc *add_out_arc(MDDNode *head, int val, int length);

    // Remove incoming arc (do not check arc tail)
    void remove_incoming(MDDArc *a);

    // Remove outgoing arc (do not check arc head)
    void remove_outgoing(MDDArc *a);
};

// DD MDDArc
//
struct MDDArc
{
    // MDDArc value
    const int val;
    // MDDNode tail
    MDDNode *tail;
    // MDDNode head
    MDDNode *head;
    // MDDArc length
    int length;
    // MDDArc index
    int index;
    // Objective weights
    ObjType *weights;

    // Constructor
    MDDArc(const int _val, MDDNode *_tail, MDDNode *_head);

    // Constructor with lengths
    MDDArc(const int _val, MDDNode *_tail, MDDNode *_head, int _length);

    // Destructor
    ~MDDArc()
    {
        delete[] weights;
    }

    // Set arc length
    void set_length(const int _length);
};


//
// Decision diagram
//
struct MDD
{
    // Number of layers
    const int num_layers;
    // List of nodes per layer
    vector<vector<MDDNode *>> layers;
    // MDD width
    int width;
    // Number of nodes
    int num_nodes;
    // Max arc value in any layer
    int max_value;
    // Longest path
    double longest_path;

    // Constructor
    MDD(const int _num_layers);

    // Destructor
    ~MDD();

    // Add a node in layer, considering max value of outgoing arcs
    MDDNode *add_node(const int layer, const int max_value);

    // Add an arc in layer
    MDDArc *add_arc(MDDNode *tail, MDDNode *head, int val);

    // Add an arc in layer considering length
    MDDArc *add_arc(MDDNode *tail, MDDNode *head, int val, int length);

    // Remove arc
    void remove(MDDArc *a);

    // Get root node
    MDDNode *get_root() const;

    // Get terminal node
    MDDNode *get_terminal() const;

    // Redirect incoming arcs from nodeB to node A
    void redirect_incoming(MDDNode *nodeA, MDDNode *nodeB);

    // Redirect existing arc as incoming to a node
    void redirect_incoming(MDDNode *node, MDDArc *in_arc);

    // Repair node indices
    void repair_node_indices();

    // Update MDD info
    void update();

    // Get MDD width (only valid when updated)
    int get_width() const;

    // Get max arc value in any layer (only valid when updated)
    int get_max_value() const;

    // Print MDD
    void print() const;

    // Check MDD consistency
    bool check_consistency();

    // Update number of nodes
    void update_num_nodes();

    // Remove nodes with zero outgoing arcs
    void remove_dangling_outgoing();

    // Remove nodes with zero incoming arcs
    void remove_dangling_incoming();

    // Remove dangling nodes
    void remove_dangling();

    // Remove node
    void remove(MDDNode *node);
};

//
// TSP MDD constructor
//
class MDDTSPConstructor
{
public:
    MDDTSPConstructor(){};
    
    // Constructor
    MDDTSPConstructor(TSPInstance *_inst);

    // Generate exact
    MDD *generate_exact();

private:
    TSPInstance *inst;
};
