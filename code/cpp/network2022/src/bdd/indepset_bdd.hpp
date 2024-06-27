// ----------------------------------------------------------
// Indepset BDD Constructor
// ----------------------------------------------------------

#ifndef INDEPSET_BDD_HPP_
#define INDEPSET_BDD_HPP_

#define BOOST_DYNAMIC_BITSET_DONT_USE_FRIENDS

#include <boost/unordered_map.hpp>
#include <boost/foreach.hpp>

#include <cassert>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <fstream>
#include <queue>

#include "../bdd/bdd.hpp"
#include "../util/util.hpp"

#include "../instances/indepset_instance.hpp"

using namespace std;

// // State definition
typedef boost::dynamic_bitset<> State;
typedef boost::unordered_map<State *, Node *, bitset_hash, bitset_equal_to> StateNodeMap;

//
// State allocator
//
struct StateAllocator
{
	// State type
	// typedef IndepSetBDDConstructor::State State;
	// Allocated states
	deque<State *> alloc_states;
	// Free states
	deque<State *> free_states;

	// Request state
	inline State *request()
	{
		if (free_states.empty())
		{
			alloc_states.push_back(new State);
			return alloc_states.back();
		}
		State *st = free_states.back();
		free_states.pop_back();
		return st;
	}

	// Deallocate state
	inline void deallocate(State *state)
	{
		free_states.push_back(state);
	}

	// // Destructor
	// ~StateAllocator()
	// {
	// 	cout << "Destructor called..." << endl;
	// 	for (deque<State *>::iterator it = alloc_states.begin(); it != alloc_states.end(); ++it)
	// 	{
	// 		delete *it;
	// 	}
	// }
	// Destructor
	void clear_states()
	{
		for (deque<State *>::iterator it = alloc_states.begin(); it != alloc_states.end(); ++it)
		{
			delete *it;
		}
	}
};

//
// Indepset BDD constructor
//
class IndepSetBDDConstructor
{
public:
	// State definition
	// typedef boost::dynamic_bitset<> State;
	// typedef boost::unordered_map<State *, Node *, bitset_hash, bitset_equal_to> StateNodeMap;

	IndepSetBDDConstructor() {}

	// Constructor
	IndepSetBDDConstructor(IndepSetInst *_inst, vector<vector<int>> &_objs);

	// Generate exact BDD
	void generate();

	bool generate_next_layer();

	// Destructor
	// ~IndepSetBDDConstructor()
	// {
	// 	delete[] in_state_counter;
	// }

	// Variable in a layer
	vector<int> var_layer;

	// Indepset instance
	IndepSetInst *inst;

	StateAllocator alloc;
	int l;

	BDD *bdd;

	bool order_provided;

	int iter, next;
	// State maps
	StateNodeMap states[2];
	// Choose next vertex in min-in-state strategy
	int choose_next_vertex_min_size_next_layer(StateNodeMap &states);

private:
	// Objectives
	vector<vector<int>> objs;
	// Number of objectives
	int num_objs;
	// Marker of the end of a state (for iteration purposes)
	int state_end;

	// Active vertices (for variable ordering)
	vector<int> active_vertices;
	// Auxiliary
	vector<pair<State, Node *>> aux_nodes;
	// Used for min-in-state variable ordering
	// int *in_state_counter;

	// void compute_states(BDD *bdd);

	int vertex;
	// weights for zero arc
	ObjType *zero_weights, *one_weights;
	StateNodeMap::iterator it;
};

// ------------------------------------------------------------------------------------------------
// Inline implementations
// ------------------------------------------------------------------------------------------------

//
// IndepsetBDD Constructor
//
inline IndepSetBDDConstructor::IndepSetBDDConstructor(IndepSetInst *_inst,
													  vector<vector<int>> &_objs)
	: inst(_inst),
	  objs(_objs),
	  num_objs(_objs.size()),
	  state_end(static_cast<int>(boost::dynamic_bitset<>::npos))
{
	// in_state_counter = new int[inst->graph->n_vertices];
	// IndepSet BDD
	bdd = new BDD(inst->graph->n_vertices + 1);

	order_provided = false;
	l = 1;
	// State maps
	iter = 0;
	next = 1;

	// initialize internal structures for variable ordering
	var_layer.resize(inst->graph->n_vertices);
	active_vertices.resize(inst->graph->n_vertices);
	// int in_state_counter[inst->graph->n_vertices];
	for (int v = 0; v < inst->graph->n_vertices; ++v)
	{
		// in_state_counter[v] = 1;
		active_vertices[v] = v;
	}

	// initialize allocator
	StateAllocator alloc = StateAllocator();

	// initialize state map
	states[iter].clear();

	// create root node
	Node *root_node = bdd->add_node(0);
	State *root_state = alloc.request();
	root_state->resize(inst->graph->n_vertices, true);
	states[iter][&root_node->setpack_state] = root_node;
	root_node->setpack_state = *root_state;

	// weights for zero arc
	zero_weights = new ObjType[NOBJS];
	memset(zero_weights, 0, sizeof(ObjType) * NOBJS);
}

//
// Choose next variable for the BDD
//
inline int IndepSetBDDConstructor::choose_next_vertex_min_size_next_layer(StateNodeMap &states)
{
	int in_state_counter[inst->graph->n_vertices];
	// update counter
	for (size_t i = 0; i < active_vertices.size(); ++i)
	{
		in_state_counter[active_vertices[i]] = 0;
	}

	BOOST_FOREACH (StateNodeMap::value_type i, states)
	{
		const State &state = *(i.first);
		for (int v = state.find_first(); v != state_end; v = state.find_next(v))
		{
			in_state_counter[v]++;
		}
	}

	int sel = 0;
	for (size_t i = 1; i < active_vertices.size(); ++i)
	{
		if (in_state_counter[active_vertices[i]] < in_state_counter[active_vertices[sel]])
		{
			sel = i;
		}
	}
	// remove vertex from active list and return it
	int v = active_vertices[sel];
	active_vertices[sel] = active_vertices.back();
	active_vertices.pop_back();

	return v;
}

#endif /* INDEPSET_BDD_HPP_ */

// ----------------------------------------------------------
