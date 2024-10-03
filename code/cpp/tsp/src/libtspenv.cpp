#include "tspenv.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(libtspenvv2, m)
{
    py::class_<TSPEnv>(m, "TSPEnv", py::module_local())
        .def(py::init<>())
        .def("reset", &TSPEnv::reset)
        .def("set_inst", &TSPEnv::set_inst)
        .def("initialize_dd_constructor", &TSPEnv::initialize_dd_constructor)
        .def("generate_dd", &TSPEnv::generate_dd)
        .def("compute_pareto_frontier", &TSPEnv::compute_pareto_frontier)
        .def("get_layer", &TSPEnv::get_layer)
        .def("get_dd", &TSPEnv::get_dd)
        .def("get_frontier", &TSPEnv::get_frontier);
}
