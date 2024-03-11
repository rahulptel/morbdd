// ----------------------------------------------------------
// Knapsack Instance
// ----------------------------------------------------------

#ifndef KNAPSACK_INSTANCE_HPP_
#define KNAPSACK_INSTANCE_HPP_

#include <vector>

using namespace std;

//
// Multiobjective knapsack problem
//
struct KnapsackInstance
{
	// Number of variables
	int n_vars;
	// Number of constraints
	int n_cons;
	// Number of objective functions
	int num_objs;
	// Objective function coefficients (indexed by variable/objective)
	vector<vector<int>> obj_coeffs;
	// Constraint coefficients (indexed by constraint/variable)
	vector<vector<int>> coeffs;
	// Right-hand sides
	vector<int> rhs;

	// Empty Constructor
	KnapsackInstance() {}

	// Direct Constructor
	KnapsackInstance(int _n_vars,
					 int _n_cons,
					 int _n_objs,
					 vector<vector<int>> _obj_coeffs,
					 vector<vector<int>> _coeffs,
					 vector<int> _rhs) : n_vars(_n_vars),
										 n_cons(_n_cons),
										 num_objs(_n_objs),
										 obj_coeffs(_obj_coeffs),
										 coeffs(_coeffs),
										 rhs(_rhs)
	{
		cout << n_vars << endl;
	}

	// Read instance based on our format
	void read(char *filename);

	// Print Instance
	void print();

	// Reorder variables based on constraint coefficients
	void reorder_coefficients();
};

#endif
