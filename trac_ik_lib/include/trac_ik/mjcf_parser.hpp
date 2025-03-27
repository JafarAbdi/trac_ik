#include <mujoco/mujoco.h>
#include <spdlog/spdlog.h>

#include <experimental/array>
#include <iostream>
#include <kdl/chain.hpp>
#include <kdl/frames.hpp>
#include <kdl/frames_io.hpp>
#include <kdl/joint.hpp>
#include <kdl/segment.hpp>
#include <kdl/tree.hpp>
#include <memory>
#include <range/v3/to_container.hpp>
#include <ranges>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

template <>
struct fmt::formatter<mjtJoint> : formatter<std::string_view> {
  auto format(const mjtJoint& joint_type, fmt::format_context& ctx) const -> fmt::format_context::iterator {
    switch (joint_type) {
      case mjtJoint::mjJNT_FREE:
        return fmt::formatter<std::string_view>::format("mjJNT_FREE", ctx);
      case mjtJoint::mjJNT_BALL:
        return fmt::formatter<std::string_view>::format("mjJNT_BALL", ctx);
      case mjtJoint::mjJNT_SLIDE:
        return fmt::formatter<std::string_view>::format("mjJNT_SLIDE", ctx);
      case mjtJoint::mjJNT_HINGE:
        return fmt::formatter<std::string_view>::format("mjJNT_HINGE", ctx);
    }
    return fmt::formatter<std::string_view>::format("Unknown", ctx);
  }
};

namespace mjcf_parser {

// Convert MuJoCo frame to KDL frame
KDL::Rotation mjToKdl(const double* mjQuat);

KDL::Frame mjToKdl(const double* mjPos, const double* mjQuat);

// construct joint
KDL::Joint toKdl(const std::string& joint_name, const mjtJoint joint_type, const mjtNum* mj_pos, const mjtNum* mj_axis);

bool treeFromMjcfModel(const mjModel* model, KDL::Tree& tree);

}  // namespace mjcf_parser
