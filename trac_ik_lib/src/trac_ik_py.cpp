#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <trac_ik/mjcf_parser.hpp>
#include <trac_ik/trac_ik.hpp>

namespace py = pybind11;

PYBIND11_MODULE(trac_ik_py, m) {
  py::enum_<TRAC_IK::SolveType>(m, "SolveType")
      .value("Speed", TRAC_IK::SolveType::Speed)
      .value("Distance", TRAC_IK::SolveType::Distance)
      .value("Manip1", TRAC_IK::SolveType::Manip1)
      .value("Manip2", TRAC_IK::SolveType::Manip2);
  py::class_<TRAC_IK::TRAC_IK>(m, "TRAC_IK")
      .def(py::init<const std::string&, const std::string&, const std::string&, double, double, TRAC_IK::SolveType>())
      .def(py::init([](const std::string& base_link,
                       const std::string& tip_link,
                       py::object model,
                       double maxtime,
                       double eps,
                       TRAC_IK::SolveType type) {
        if (mjVERSION_HEADER != mj_version()) {
          throw std::runtime_error(
              fmt::format("MuJoCo library and header mismatch! mjVERSION_HEADER: {} mj_version(): {}",
                          mjVERSION_HEADER,
                          mj_version()));
        }
        std::uintptr_t m_raw = model.attr("_address").cast<std::uintptr_t>();
        const mjModel* mj_model = reinterpret_cast<const mjModel*>(m_raw);
        return std::make_unique<TRAC_IK::TRAC_IK>(base_link, tip_link, mj_model, maxtime, eps, type);
      }))
      .def("getNrOfSegments",
           [](TRAC_IK::TRAC_IK& self) {
             KDL::Chain chain;
             self.getKDLChain(chain);
             return chain.getNrOfSegments();
           })
      .def("CartToJnt",
           [](TRAC_IK::TRAC_IK& self,
              const std::vector<double>& q_init,
              const std::array<double, 7>& pose,
              const std::array<double, 6>& bounds = {}) {
             // pose Uses mujoco convention x y z rw rx ry rz
             // bounds x y z rx ry rz
             const auto frame = mjcf_parser::mjToKdl(pose.data(), pose.data() + 3);
             KDL::JntArray in(q_init.size());
             KDL::JntArray out(q_init.size());
             for (uint z = 0; z < q_init.size(); z++) in(z) = q_init[z];

             KDL::Twist kdl_bounds = KDL::Twist::Zero();
             kdl_bounds.vel.x(bounds[0]);
             kdl_bounds.vel.y(bounds[1]);
             kdl_bounds.vel.z(bounds[2]);
             kdl_bounds.rot.x(bounds[3]);
             kdl_bounds.rot.y(bounds[4]);
             kdl_bounds.rot.z(bounds[5]);
             int rc = self.CartToJnt(in, frame, out, kdl_bounds);
             std::vector<double> vout;
             // If no solution, return empty vector which acts as None
             if (rc == -3) return vout;

             for (uint z = 0; z < q_init.size(); z++) vout.push_back(out(z));

             return vout;
           })
      .def("getNrOfJointsInChain",
           [](TRAC_IK::TRAC_IK& self) {
             KDL::Chain chain;
             self.getKDLChain(chain);
             return chain.getNrOfJoints();
           })
      .def("getJointNamesInChain",
           [](TRAC_IK::TRAC_IK& self) {
             KDL::Chain chain;
             self.getKDLChain(chain);
             std::vector<std::string> joint_names;
             for (const auto& segment : chain.segments) {
               if (segment.getJoint().getType() == KDL::Joint::JointType::None) continue;
               joint_names.push_back(segment.getJoint().getName());
             }
             return joint_names;
           })
      .def("getLinkNamesInChain",
           [](TRAC_IK::TRAC_IK& self) {
             KDL::Chain chain;
             self.getKDLChain(chain);
             std::vector<std::string> link_names;
             link_names.reserve(chain.segments.size());
             for (const auto& segment : chain.segments) {
               link_names.push_back(segment.getName());
             }
             return link_names;
           })
      .def("getLowerBoundLimits",
           [](TRAC_IK::TRAC_IK& self) {
             KDL::JntArray lb;
             KDL::JntArray ub;
             self.getKDLLimits(lb, ub);
             std::vector<double> lb_vec;
             lb_vec.reserve(lb.rows());
             for (int i = 0; i < lb.rows(); i++) {
               lb_vec.push_back(lb(i));
             }
             return lb_vec;
           })
      .def("getUpperBoundLimits",
           [](TRAC_IK::TRAC_IK& self) {
             KDL::JntArray lb;
             KDL::JntArray ub;
             self.getKDLLimits(lb, ub);
             std::vector<double> ub_vec;
             ub_vec.reserve(ub.rows());
             for (int i = 0; i < ub.rows(); i++) {
               ub_vec.push_back(ub(i));
             }
             return ub_vec;
           })
      .def("JntToCart",
           [](TRAC_IK::TRAC_IK& self, const std::vector<double>& q) {
             KDL::JntArray in(q.size());
             for (uint z = 0; z < q.size(); z++) in(z) = q[z];
             const auto frame = self.JntToCart(in);
             double rw;
             double rx;
             double ry;
             double rz;
             frame.M.GetQuaternion(rx, ry, rz, rw);
             return std::array<double, 7>{frame.p.x(), frame.p.y(), frame.p.z(), rw, rx, ry, rz};
           })
      .def("JntToCartFrames", [](TRAC_IK::TRAC_IK& self, const std::vector<double>& q) {
        KDL::JntArray in(q.size());
        for (uint z = 0; z < q.size(); z++) in(z) = q[z];
        const auto frames = self.JntToCartFrames(in);
        std::vector<std::tuple<std::string, std::string, std::array<double, 7>>> out(frames.size());
        out.reserve(frames.size());
        KDL::Chain chain;
        self.getKDLChain(chain);
        for (size_t i = 0; i < frames.size(); i++) {
          double rw;
          double rx;
          double ry;
          double rz;
          const auto& frame = frames[i];
          frame.M.GetQuaternion(rx, ry, rz, rw);
          const auto& segment = chain.getSegment(i);
          out.emplace_back(segment.getName(),
                           segment.getJoint().getName(),
                           std::experimental::make_array(frame.p.x(), frame.p.y(), frame.p.z(), rw, rx, ry, rz));
        }
        return out;
      });
}
