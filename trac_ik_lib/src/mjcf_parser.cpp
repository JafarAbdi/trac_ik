#include <fmt/ranges.h>

#include <span>
#include <trac_ik/mjcf_parser.hpp>
void print_tree_element(const KDL::TreeElement& tree_element) {
  const auto& segment = tree_element.segment;
  spdlog::info("Segment '{}' with joint '{}' at origin {} and tip at {} with rotation {}",
               segment.getName(),
               segment.getJoint().getName(),
               segment.getJoint().JointOrigin().data,
               segment.getFrameToTip().p.data,
               segment.getFrameToTip().M.data);
  spdlog::info("Segment '{}' has {} children.", segment.getName(), tree_element.children.size());
  for (const auto& child : tree_element.children) {
    print_tree_element(child->second);
  }
}

bool mjcf_parser::treeFromMjcfModel(const mjModel* model, KDL::Tree& tree) {
  if (model->body_parentid[0] != 0) {
    spdlog::error("Root body has parent ID {} - Expected 0.", model->body_parentid[0]);
    return false;
  }

  const std::string root_name = mj_id2name(model, mjtObj::mjOBJ_BODY, 0);
  tree = KDL::Tree(root_name);
  tree.addSegment(KDL::Segment(root_name, KDL::Joint(KDL::Joint::Fixed)), "root");
  spdlog::info("Added segment '{}' to tree with parent '{}'.", root_name, "root");
  for (int body_id = 1; body_id < model->nbody; body_id++) {
    const auto parent_id = model->body_parentid[body_id];
    const std::string parent_body_name = mj_id2name(model, mjtObj::mjOBJ_BODY, parent_id);
    const std::string body_name = mj_id2name(model, mjtObj::mjOBJ_BODY, body_id);
    const auto number_of_joints = model->body_jntnum[body_id];
    if (number_of_joints > 1) {
      spdlog::warn("Body '{}' has {} joints. Only the first joint will be considered.", body_name, number_of_joints);
    }
    const auto frame = mjToKdl(&model->body_pos[3 * body_id], &model->body_quat[4 * body_id]);
    if (number_of_joints == 0) {
      if (!tree.addSegment(KDL::Segment(body_name, KDL::Joint(KDL::Joint::Fixed), frame), parent_body_name)) {
        spdlog::error("Failed to add segment '{}' to tree with parent '{}'.", body_name, parent_body_name);
        return false;
      }
    } else {
      const auto joint_id = model->body_jntadr[body_id];
      const auto joint_name = mj_id2name(model, mjtObj::mjOBJ_JOINT, joint_id);
      const auto joint = toKdl(joint_name,
                               mjtJoint(model->jnt_type[joint_id]),
                               &model->jnt_pos[3 * joint_id],
                               &model->jnt_axis[3 * joint_id]);
      spdlog::info("{} -> {} -> {}", parent_body_name, joint_name, body_name);
      auto segment = KDL::Segment(body_name, joint, frame);
      if (!tree.addSegment(segment, parent_body_name)) {
        spdlog::error("Failed to add segment '{}' with joint '{}'.", body_name, joint_name);
        return false;
      }
    }
  }
  // print_tree_element(tree.getRootSegment()->second);
  return true;
}

KDL::Joint mjcf_parser::toKdl(const std::string& joint_name,
                              const mjtJoint joint_type,
                              const mjtNum* mj_pos,
                              const mjtNum* mj_axis) {
  const auto origin = KDL::Vector(mj_pos[0], mj_pos[1], mj_pos[2]);
  const auto axis = KDL::Vector(mj_axis[0], mj_axis[1], mj_axis[2]);
  if (joint_type == mjtJoint::mjJNT_HINGE) {
    return KDL::Joint(joint_name, origin, axis, KDL::Joint::RotAxis);
  }
  if (joint_type == mjtJoint::mjJNT_SLIDE) {
    return KDL::Joint(joint_name, origin, axis, KDL::Joint::TransAxis);
  }
  spdlog::warn("Converting unknown joint type '{}' of joint '{}' into a fixed joint", joint_type, joint_name);
  return KDL::Joint(joint_name, KDL::Joint::None);
}

KDL::Frame mjcf_parser::mjToKdl(const double* mjPos, const double* mjQuat) {
  KDL::Frame kdl_frame;

  kdl_frame.p.x(mjPos[0]);
  kdl_frame.p.y(mjPos[1]);
  kdl_frame.p.z(mjPos[2]);
  kdl_frame.M = mjToKdl(mjQuat);

  return kdl_frame;
}

KDL::Rotation mjcf_parser::mjToKdl(const double* mjQuat) {
  return KDL::Rotation::Quaternion(mjQuat[1], mjQuat[2], mjQuat[3], mjQuat[0]);
}
