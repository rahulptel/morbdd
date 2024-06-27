// ----------------------------------------------------------
// Indepset BDD Constructor - Implementation
// ----------------------------------------------------------

#include <deque>

#include "indepset_bdd.hpp"
#include "bdd_alg.hpp"

#include "../util/util.hpp"

using namespace boost;

//
// Generate next layer in BDD
//
bool IndepSetBDDConstructor::generate_next_layer()
{
	// If the last layer is approximated update the states[iter]
	if (states[iter].size() > bdd->layers[l - 1].size())
	{
		states[iter].clear();
		for (int k = 0; k < bdd->layers[l - 1].size(); ++k)
		{
			states[iter][&bdd->layers[l - 1][k]->setpack_state] = bdd->layers[l - 1][k];
		}
	}

	// cout << "\nCreating IndepSet BDD..." << endl;
	if (l < inst->graph->n_vertices + 1)
	{
		states[next].clear();
		// select next vertex
		vertex = var_layer[l - 1];

		// set weights for one arc
		one_weights = new ObjType[NOBJS];
		for (int p = 0; p < NOBJS; ++p)
		{
			one_weights[p] = objs[p][vertex];
		}

		// cout << "\tLayer " << l << " - vertex=" << vertex << " - size=" << states[iter].size() << '\n';

		BOOST_FOREACH (StateNodeMap::value_type i, states[iter])
		{
			State state = *(i.first);
			Node *node = i.second;
			bool was_set = state[vertex];

			// zero arc
			state.set(vertex, false);
			it = states[next].find(&state);
			if (it == states[next].end())
			{
				Node *new_node = bdd->add_node(l);
				// State *new_state = alloc.request();
				// (*new_state) = state;
				// states[next][new_state] = new_node;
				new_node->setpack_state = state;
				states[next][&new_node->setpack_state] = new_node;

				node->add_out_arc(new_node, 0);
				node->set_arc_weights(0, zero_weights);
			}
			else
			{
				node->add_out_arc(it->second, 0);
				node->set_arc_weights(0, zero_weights);
			}

			// one arc
			if (was_set)
			{
				state &= inst->adj_mask_compl[vertex];
				it = states[next].find(&state);
				if (it == states[next].end())
				{
					Node *new_node = bdd->add_node(l);
					// State *new_state = alloc.request();
					// (*new_state) = state;
					new_node->setpack_state = state;
					states[next][&new_node->setpack_state] = new_node;

					node->add_out_arc(new_node, 1);
					node->set_arc_weights(1, one_weights);
				}
				else
				{
					node->add_out_arc(it->second, 1);
					node->set_arc_weights(1, one_weights);
				}
			}

			// deallocate node state
			// alloc.deallocate(i.first);
		}

		// invert iter and next
		next = !next;
		iter = !iter;

		++l;
		if (l < inst->graph->n_vertices + 1)
		{
			return false;
		}
	}
	return true;
}

//
// Create BDD
//
void IndepSetBDDConstructor::generate()
{
	// cout << "\nCreating IndepSet BDD..." << endl;
	l = 1;
	bool is_done;
	do
	{
		set_var_layer();
		is_done = generate_next_layer();
	} while (!is_done);
}
