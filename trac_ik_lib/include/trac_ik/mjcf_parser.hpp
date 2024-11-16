#include <kdl/joint.hpp>
#include <kdl/tree.hpp>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>

#include <kdl/frames_io.hpp>

#include <mujoco/mujoco.h>
#include <spdlog/spdlog.h>

#include <experimental/array>
#include <iostream>
#include <memory>
#include <range/v3/to_container.hpp>
#include <ranges>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include <kdl/segment.hpp>
#include <kdl/tree.hpp>
#include <spdlog/spdlog.h>

#include <kdl/frames.hpp>
#include <kdl/segment.hpp>
#include <kdl/tree.hpp>
#include <spdlog/spdlog.h>

#include <kdl/chain.hpp>
#include <kdl/frames.hpp>
#include <kdl/joint.hpp>
#include <kdl/segment.hpp>
#include <kdl/tree.hpp>
#include <spdlog/spdlog.h>
#include <string>
#include <vector>

#include <kdl/chain.hpp>
#include <kdl/frames.hpp>
#include <kdl/joint.hpp>
#include <kdl/segment.hpp>
#include <kdl/tree.hpp>
#include <spdlog/spdlog.h>
#include <string>
#include <vector>

template <> struct fmt::formatter<mjtJoint> : formatter<std::string_view> {
  auto format(const mjtJoint &joint_type, fmt::format_context &ctx) const
      -> fmt::format_context::iterator {
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
// Convert MuJoCo frame to KDL frame
KDL::Rotation mjToKdl(const double *mjQuat) {
  return KDL::Rotation::Quaternion(mjQuat[1], mjQuat[2], mjQuat[3], mjQuat[0]);
}

KDL::Frame mjToKdl(const double *mjPos, const double *mjQuat) {
  KDL::Frame kdlFrame;

  kdlFrame.p.x(mjPos[0]);
  kdlFrame.p.y(mjPos[1]);
  kdlFrame.p.z(mjPos[2]);
  kdlFrame.M = mjToKdl(mjQuat);

  return kdlFrame;
}

struct JointInfo {
  int id;                      // Joint ID in mjModel
  std::string name;            // Joint name
  mjtJoint type;               // Joint type (mjtJoint)
  bool limited;                // Joint is limited
  std::array<mjtNum, 2> range; // Joint limits
  std::array<mjtNum, 3> axis;  // Joint axis
  std::array<mjtNum, 3> pos;
};

struct BodyNode {
  int id;                     // Body ID in mjModel
  std::string name;           // Body name
  int parent_id;              // Parent body ID
  std::vector<int> child_ids; // Child body IDs
  std::array<mjtNum, 3> pos;  // Position relative to parent
  std::array<mjtNum, 4> quat; // Orientation relative to parent
  std::optional<JointInfo> joint;
};

auto buildTree(const mjModel *model) {
  std::vector<BodyNode> bodies_ =
      std::views::iota(0, model->nbody) |
      std::views::transform([model](const auto body_idx) {
        return BodyNode{.id = body_idx,
                        .name = mj_id2name(model, mjtObj::mjOBJ_BODY, body_idx),
                        .parent_id = model->body_parentid[body_idx],
                        .pos = std::experimental::make_array(
                            model->body_pos[3 * body_idx],
                            model->body_pos[3 * body_idx + 1],
                            model->body_pos[3 * body_idx + 2]),
                        .quat = std::experimental::make_array(
                            model->body_quat[4 * body_idx],
                            model->body_quat[4 * body_idx + 1],
                            model->body_quat[4 * body_idx + 2],
                            model->body_quat[4 * body_idx + 3]),
                        .joint = std::nullopt};
      }) |
      ranges::to_vector;

  // Build child relationships
  // Start from 1 to skip world body (id=0 has parent_id=0)
  for (int body_idx = 1; body_idx < model->nbody; ++body_idx) {
    bodies_[model->body_parentid[body_idx]].child_ids.push_back(body_idx);
  }
  // Build joint relationships
  for (int body_idx = 0; body_idx < model->nbody; ++body_idx) {
    const auto parent_id = model->body_parentid[body_idx];
    if (model->body_jntadr[parent_id] != -1) {
      const auto number_joints = model->body_jntnum[parent_id];
      spdlog::debug("Body {} has {} joints", bodies_[parent_id].name,
                    number_joints);
      for (auto joint_idx = 0; joint_idx < number_joints; ++joint_idx) {
        const auto joint_id = model->body_jntadr[parent_id] + joint_idx;
        if (model->jnt_bodyid[joint_id] == parent_id) {
          bodies_[body_idx].joint = JointInfo{
              .id = joint_id,
              .name = mj_id2name(model, mjtObj::mjOBJ_JOINT, joint_id),
              .type = mjtJoint(model->jnt_type[joint_id]),
              .limited = static_cast<bool>(model->jnt_limited[joint_id]),
              .range = std::experimental::make_array(
                  model->jnt_range[2 * joint_id],
                  model->jnt_range[2 * joint_id + 1]),
              .axis = std::experimental::make_array(
                  model->jnt_axis[3 * joint_id],
                  model->jnt_axis[3 * joint_id + 1],
                  model->jnt_axis[3 * joint_id + 2]),
              .pos = std::experimental::make_array(
                  model->jnt_pos[3 * joint_id],
                  model->jnt_pos[3 * joint_id + 1],
                  model->jnt_pos[3 * joint_id + 2]),
          };
          spdlog::debug("Joint {} [parent: {} - child {}]",
                        bodies_[body_idx].joint.value().name,
                        bodies_[parent_id].name, bodies_[body_idx].name);
        }
      }
    }
  }
  return bodies_;
}

namespace mjcf_parser {

// construct joint
KDL::Joint toKdl(const JointInfo &joint) {

  const auto origin = KDL::Vector(joint.pos[0], joint.pos[1], joint.pos[2]);
  const auto axis = KDL::Vector(joint.axis[0], joint.axis[1], joint.axis[2]);
  if (joint.type == mjtJoint::mjJNT_HINGE) {
    return KDL::Joint(joint.name, origin, axis, KDL::Joint::RotAxis);
  } else if (joint.type == mjtJoint::mjJNT_SLIDE) {
    return KDL::Joint(joint.name, origin, axis, KDL::Joint::TransAxis);
  }
  spdlog::warn(
      "Converting unknown joint type '{}' of joint '{}' into a fixed joint",
      joint.type, joint.name);
  return KDL::Joint(joint.name, KDL::Joint::None);
}

void addSegment(KDL::Tree &tree, const BodyNode &root,
                const std::string &parent_name, const KDL::Joint &joint) {
  spdlog::debug("Adding segment {} to parent {}", root.name, parent_name);
  // construct the kdl segment
  KDL::Segment sgm(root.name, joint,
                   mjToKdl(root.pos.data(), root.quat.data()));

  // add segment to tree
  tree.addSegment(sgm, parent_name);
}

// recursive function to walk through tree
bool addChildrenToTree(const BodyNode &root,
                       const std::vector<BodyNode> &kinematics_tree,
                       KDL::Tree &tree) {
  if (root.joint.has_value()) {
    addSegment(tree, root, kinematics_tree.at(root.parent_id).name,
               toKdl(root.joint.value()));
  } else {
    addSegment(tree, root, kinematics_tree.at(root.parent_id).name,
               KDL::Joint(KDL::Joint::Fixed));
  }

  const auto &children = root.child_ids;
  spdlog::debug("Link {} had {} children", root.name, children.size());

  // recurslively add all children
  for (size_t i = 0; i < children.size(); i++) {
    if (!addChildrenToTree(kinematics_tree.at(children[i]), kinematics_tree,
                           tree)) {
      return false;
    }
  }
  return true;
}

bool treeFromMjcfModel(const mjModel *model, KDL::Tree &tree) {
  const auto kinematics_tree = buildTree(model);
  tree = KDL::Tree(kinematics_tree[0].name);

  //  add all children
  for (size_t i = 0; i < kinematics_tree[0].child_ids.size(); i++) {
    if (!addChildrenToTree(kinematics_tree.at(i), kinematics_tree, tree)) {
      return false;
    }
  }
  return true;
}

} // namespace mjcf_parser
