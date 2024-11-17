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

struct JointInfo {
  int id;                       // Joint ID in mjModel
  std::string name;             // Joint name
  mjtJoint type;                // Joint type (mjtJoint)
  bool limited;                 // Joint is limited
  std::array<mjtNum, 2> range;  // Joint limits
  std::array<mjtNum, 3> axis;   // Joint axis
  std::array<mjtNum, 3> pos;
};

struct BodyNode {
  int id;                      // Body ID in mjModel
  std::string name;            // Body name
  int parent_id;               // Parent body ID
  std::vector<int> child_ids;  // Child body IDs
  std::array<mjtNum, 3> pos;   // Position relative to parent
  std::array<mjtNum, 4> quat;  // Orientation relative to parent
  std::optional<JointInfo> joint;
};

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

std::vector<BodyNode> buildTree(const mjModel* model);

// construct joint
KDL::Joint toKdl(const JointInfo& joint);

void addSegment(KDL::Tree& tree, const BodyNode& root, const std::string& parent_name, const KDL::Joint& joint);

// recursive function to walk through tree
bool addChildrenToTree(const BodyNode& root, const std::vector<BodyNode>& kinematics_tree, KDL::Tree& tree);

bool treeFromMjcfModel(const mjModel* model, KDL::Tree& tree);

}  // namespace mjcf_parser
