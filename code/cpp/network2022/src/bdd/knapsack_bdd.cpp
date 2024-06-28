// ----------------------------------------------------------
// Knapsack BDD Constructor - Implementations
// ----------------------------------------------------------

#include "knapsack_bdd.hpp"

//
// Generate next layer
//
bool KnapsackBDDConstructor::generate_next_layer()
{

	if (l < inst->n_vars)
	{
		// cout << "\tLayer " << l << " - number of nodes: " << states[next].size() << endl;

		// Initialize one-arc weights
		ObjType *one_weights = new ObjType[NOBJS];
		for (int p = 0; p < NOBJS; ++p)
		{
			one_weights[p] = inst->obj_coeffs[l][p];
		}

		states[next].clear();
		BOOST_FOREACH (StateNodeMap::value_type i, states[iter])
		{
			// Obtain node and states
			Node *node = i.second;
			state = i.first;

			if (l < inst->n_vars - 1)
			{
				// zero arc
				StateNodeMap::iterator it = states[next].find(state);
				if (it == states[next].end())
				{
					Node *new_node = bdd->add_node(l + 1);
					new_node->weight = state;
					states[next][state] = new_node;
					node->add_out_arc_fast(new_node, 0);
					node->set_arc_weights(0, zero_weights);
				}
				else
				{
					node->add_out_arc_fast(it->second, 0);
					node->set_arc_weights(0, zero_weights);
				}

				// one arc

				// update state
				feasible = true;
				for (int c = 0; c < inst->n_cons && feasible; ++c)
				{
					state[c] += inst->coeffs[c][l];
					feasible = (state[c] <= inst->rhs[c]);
				}
				if (feasible)
				{
					StateNodeMap::iterator it = states[next].find(state);
					if (it == states[next].end())
					{
						Node *new_node = bdd->add_node(l + 1);
						new_node->weight = state;
						states[next][state] = new_node;
						node->add_out_arc_fast(new_node, 1);
						node->set_arc_weights(1, one_weights);
					}
					else
					{
						node->add_out_arc_fast(it->second, 1);
						node->set_arc_weights(1, one_weights);
					}
				}
			}
			else
			{
				// if last layer, just add arcs to the terminal node

				// zero arc
				node->add_out_arc_fast(terminal_node, 0);
				node->set_arc_weights(0, zero_weights);

				// one arc
				feasible = true;
				for (int c = 0; c < inst->n_cons && feasible; ++c)
				{
					feasible = (i.first[c] + inst->coeffs[c][l] <= inst->rhs[c]);
				}
				if (feasible)
				{
					node->add_out_arc_fast(terminal_node, 1);
					node->set_arc_weights(1, one_weights);
				}
			}
		}

		// invert iter and next
		next = !next;
		iter = !iter;
		++l;

		if (l < inst->n_vars)
		{
			return false;
		}

		// cout << "\n\tupdating incoming arcs..." << endl;
		bdd->update_incoming_arcsets();

		// Fix indices
		bdd->fix_indices();
	}

	// Non-last layer
	return true;
}

//
// Generate exact BDD
//
void KnapsackBDDConstructor::generate_exact()
{
	l = 0;
	bool is_done;
	do
	{
		is_done = generate_next_layer();
	} while (!is_done);

	// for (l = 0; l < inst->n_vars; ++l)
	// {
	// 	// cout << "\tLayer " << l << " - number of nodes: " << states[next].size() << endl;

	// 	// Initialize one-arc weights
	// 	ObjType *one_weights = new ObjType[NOBJS];
	// 	for (int p = 0; p < NOBJS; ++p)
	// 	{
	// 		one_weights[p] = inst->obj_coeffs[l][p];
	// 	}

	// 	states[next].clear();
	// 	BOOST_FOREACH (StateNodeMap::value_type i, states[iter])
	// 	{
	// 		// Obtain node and states
	// 		Node *node = i.second;
	// 		state = i.first;

	// 		if (l < inst->n_vars - 1)
	// 		{
	// 			// zero arc
	// 			StateNodeMap::iterator it = states[next].find(state);
	// 			if (it == states[next].end())
	// 			{
	// 				Node *new_node = bdd->add_node(l + 1);
	//              new_node->weight = state;
	// 				states[next][state] = new_node;
	// 				node->add_out_arc_fast(new_node, 0);
	// 				node->set_arc_weights(0, zero_weights);
	// 			}
	// 			else
	// 			{
	// 				node->add_out_arc_fast(it->second, 0);
	// 				node->set_arc_weights(0, zero_weights);
	// 			}

	// 			// one arc

	// 			// update state
	// 			feasible = true;
	// 			for (int c = 0; c < inst->n_cons && feasible; ++c)
	// 			{
	// 				state[c] += inst->coeffs[c][l];
	// 				feasible = (state[c] <= inst->rhs[c]);
	// 			}
	// 			if (feasible)
	// 			{
	// 				StateNodeMap::iterator it = states[next].find(state);
	// 				if (it == states[next].end())
	// 				{
	// 					Node *new_node = bdd->add_node(l + 1);
	// 					new_node->weight = state;
	// 					states[next][state] = new_node;

	// 					node->add_out_arc_fast(new_node, 1);
	// 					node->set_arc_weights(1, one_weights);
	// 				}
	// 				else
	// 				{
	// 					node->add_out_arc_fast(it->second, 1);
	// 					node->set_arc_weights(1, one_weights);
	// 				}
	// 			}
	// 		}
	// 		else
	// 		{
	// 			// if last layer, just add arcs to the terminal node

	// 			// zero arc
	// 			node->add_out_arc_fast(terminal_node, 0);
	// 			node->set_arc_weights(0, zero_weights);

	// 			// one arc
	// 			feasible = true;
	// 			for (int c = 0; c < inst->n_cons && feasible; ++c)
	// 			{
	// 				feasible = (i.first[c] + inst->coeffs[c][l] <= inst->rhs[c]);
	// 			}
	// 			if (feasible)
	// 			{
	// 				node->add_out_arc_fast(terminal_node, 1);
	// 				node->set_arc_weights(1, one_weights);
	// 			}
	// 		}
	// 	}

	// 	// invert iter and next
	// 	next = !next;
	// 	iter = !iter;
	// }
	// cout << "\n\tupdating incoming arcs..." << endl;
	// bdd->update_incoming_arcsets();

	// // Fix indices
	// bdd->fix_indices();

	// // cout << "\tdone" << endl;
	// // return bdd;
}

//
// Update node weights
//
void KnapsackBDDConstructor::update_node_weights(BDD *bdd)
{
	// Run top-down to compute the minimum node weights (i.e., the used capacity)
	// cout << endl << "Updating node weights..." << endl;

	// root node
	bdd->get_root()->min_weight = 0;

	for (int l = 1; l < bdd->num_layers; ++l)
	{
		// iterate on layers
		for (vector<Node *>::iterator it = bdd->layers[l].begin(); it != bdd->layers[l].end(); ++it)
		{
			(*it)->min_weight = INT_MAX;

			// iterate over the incoming zero arcs
			for (vector<Node *>::iterator it_prev = (*it)->prev[0].begin(); it_prev != (*it)->prev[0].end(); ++it_prev)
			{
				(*it)->min_weight = min((*it)->min_weight, (*it_prev)->min_weight);
			}

			// iterate over the incoming one arcs
			for (vector<Node *>::iterator it_prev = (*it)->prev[1].begin(); it_prev != (*it)->prev[1].end(); ++it_prev)
			{
				(*it)->min_weight = min((*it)->min_weight, (*it_prev)->min_weight + inst->coeffs[0][l - 1]);
			}
		}
	}
}

void KnapsackBDDConstructor::fix_state_map()
{
	// If the last layer is approximated update the states[iter]
	// We insert nodes in layer l+1. Hence, l is the last layer.
	if (states[iter].size() > bdd->layers[l].size())
	{
		states[iter].clear();
		for (int k = 0; k < bdd->layers[l].size(); ++k)
		{

			states[iter][bdd->layers[l][k]->weight] = bdd->layers[l][k];
		}
	}
}