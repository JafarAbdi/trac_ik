#include <fmt/ranges.h>

#include <span>
#include <trac_ik/mjcf_parser.hpp>

bool mjcf_parser::treeFromMjcfModel(const mjModel* model, KDL::Tree& tree) {
  // id 0 corresponds to the world body
  const std::string root_name = mj_id2name(model, mjtObj::mjOBJ_BODY, 0);
  tree = KDL::Tree(root_name);
  tree.addSegment(KDL::Segment(root_name, KDL::Joint(KDL::Joint::Fixed)), "root");
  for (int body_id = 1; body_id < model->nbody; body_id++) {
    const auto parent_id = model->body_parentid[body_id];
    const auto parent_body_name = mj_id2name(model, mjtObj::mjOBJ_BODY, parent_id);
    const auto body_name = mj_id2name(model, mjtObj::mjOBJ_BODY, body_id);
    const auto number_of_joints = model->body_jntnum[parent_id];
    const auto joint_id = model->body_jntadr[parent_id];
    if (number_of_joints > 1) {
      spdlog::warn("Body '{}' has {} joints. Only the first joint will be considered.", body_name, number_of_joints);
    }
    const auto frame = mjToKdl(&model->body_pos[3 * body_id], &model->body_quat[4 * body_id]);
    auto joint = KDL::Joint(KDL::Joint::Fixed);
    if (number_of_joints != 0) {
      const auto joint_name = mj_id2name(model, mjtObj::mjOBJ_JOINT, joint_id);
      joint = toKdl(joint_name,
                    mjtJoint(model->jnt_type[joint_id]),
                    &model->jnt_pos[3 * joint_id],
                    &model->jnt_axis[3 * joint_id]);
    }
    spdlog::debug("{} -> {} -> {}", parent_body_name, joint.getName(), body_name);
    auto segment = KDL::Segment(body_name, joint, frame);
    if (!tree.addSegment(segment, parent_body_name)) {
      spdlog::error("Failed to add segment '{}' with joint '{}'.", body_name, joint.getName());
      return false;
    }
  }
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
