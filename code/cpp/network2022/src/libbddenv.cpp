#include "bddenv.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(libbddenvv2o5, m)
{
    py::class_<BDDEnv>(m, "BDDEnv")
        .def(py::init<>())
        .def("reset", &BDDEnv::reset)
        .def("set_inst", &BDDEnv::set_inst)
        .def("preprocess_inst", &BDDEnv::preprocess_inst)
        .def("initialize_dd_constructor", &BDDEnv::initialize_dd_constructor)
        .def("generate_dd", &BDDEnv::generate_dd)
        .def("generate_next_layer", &BDDEnv::generate_next_layer)
        .def("approximate_layer", &BDDEnv::approximate_layer)
        .def("get_dd", &BDDEnv::get_dd)
        .def("get_layer", &BDDEnv::get_layer)
        .def("reduce_dd", &BDDEnv::reduce_dd)
        .def("compute_pareto_frontier", &BDDEnv::compute_pareto_frontier)
        .def("get_var_layer", &BDDEnv::get_var_layer)
        .def("get_frontier", &BDDEnv::get_frontier)
        .def("get_time", &BDDEnv::get_time)
        .def("get_num_nodes_per_layer", &BDDEnv::get_num_nodes_per_layer)
        .def_readwrite("initial_width", &BDDEnv::initial_width)
        .def_readwrite("initial_node_count", &BDDEnv::initial_node_count)
        .def_readwrite("initial_arcs_count", &BDDEnv::initial_arcs_count)
        .def_readwrite("reduced_width", &BDDEnv::reduced_width)
        .def_readwrite("reduced_node_count", &BDDEnv::reduced_node_count)
        .def_readwrite("reduced_arcs_count", &BDDEnv::reduced_arcs_count)
        .def_readwrite("max_in_degree_per_layer", &BDDEnv::max_in_degree_per_layer)
        .def_readwrite("initial_num_nodes_per_layer", &BDDEnv::initial_num_nodes_per_layer)
        .def_readwrite("reduced_num_nodes_per_layer", &BDDEnv::reduced_num_nodes_per_layer)
        .def_readwrite("num_pareto_sol_per_layer", &BDDEnv::num_pareto_sol_per_layer)
        .def_readwrite("num_comparisons_per_layer", &BDDEnv::num_comparisons_per_layer)
        .def_readwrite("in_degree", &BDDEnv::in_degree)
        .def_readwrite("nnds", &BDDEnv::nnds)
        .def_readwrite("z_sol", &BDDEnv::z_sol);
}
