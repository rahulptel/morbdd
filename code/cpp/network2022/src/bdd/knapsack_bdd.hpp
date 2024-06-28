// ----------------------------------------------------------
// Knapsack BDD Constructor
// ----------------------------------------------------------

#ifndef KNAPSACK_BDD_HPP_
#define KNAPSACK_BDD_HPP_

#include <boost/unordered_map.hpp>
#include <boost/foreach.hpp>

#include "bdd.hpp"
#include "../instances/knapsack_instance.hpp"

//
// Knapsack BDD constructor
//
class KnapsackBDDConstructor
{
public:
	// State definitions
	typedef vector<int> State;
	typedef boost::unordered_map<State, Node *> StateNodeMap;

	// Constructor
	KnapsackBDDConstructor();

	KnapsackBDDConstructor(KnapsackInstance *);

	// Generate exact BDD
	void generate_exact();

	// Generate next layer
	bool generate_next_layer();
	void fix_state_map();

	// Update node weights
	void update_node_weights(BDD *bdd);

	// BDD
	BDD *bdd;

	// State maps
	StateNodeMap states[2];
	int l;
	int iter, next;

private:
	// Knapsack Instance
	KnapsackInstance *inst;

	State state;
	Node *root_node;
	Node *terminal_node;
	bool feasible;
	ObjType *zero_weights;
};

inline KnapsackBDDConstructor::KnapsackBDDConstructor()
{
}

inline KnapsackBDDConstructor::KnapsackBDDConstructor(KnapsackInstance *_inst)
	: inst(_inst)
{
	cout << inst->obj_coeffs[0][0] << endl;
	bdd = new BDD(inst->n_vars + 1);

	// Layer
	l = 0;

	// State maps
	iter = 0;
	next = 1;

	// State information
	state = State(inst->n_cons, 0);

	// create root node
	root_node = bdd->add_node(0);
	states[iter].clear();
	states[iter][state] = root_node;
	root_node->weight = state;

	// create terminal node
	terminal_node = bdd->add_node(inst->n_vars);

	// auxiliaries
	feasible = false;

	// Zero-arc weights
	zero_weights = new ObjType[NOBJS];
	memset(zero_weights, 0, sizeof(ObjType) * NOBJS);
}

#endif