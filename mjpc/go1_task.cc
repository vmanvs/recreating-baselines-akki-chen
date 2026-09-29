#include "go1_task.h"
#include <algorithm>
#include <cmath>
#include "mjpc/utilities.h"

namespace go1 {
Task::Task() {
  num_residual = num_term = num_trace = 0;
  risk = 0;
  residual_ = std::make_unique<ResidualFn>(this);
}

void Task::ResetLocked(const mjModel* m) {
  torso_ = mj_name2id(m, mjOBJ_BODY, "trunk");
  floor_ = mj_name2id(m, mjOBJ_GEOM, "floor");
  home_ = mj_name2id(m, mjOBJ_KEY, "home");
  velocity_adr_ = m->sensor_adr[mj_name2id(m, mjOBJ_SENSOR, "local_linvel")];
  const char* feet[] = {"FR", "FL", "RR", "RL"};
  const double* offsets = mjpc::GetCustomNumericData(m, "go1_gait_offsets");
  gait_startup_s_ = mjpc::GetNumberOrDefault(0.0, m, "go1_gait_startup");
  for (int j = 0; j < 4; ++j) {
    foot_site_[j] = mj_name2id(m, mjOBJ_SITE, feet[j]);
    foot_geom_[j] = mj_name2id(m, mjOBJ_GEOM, feet[j]);
    calf_body_[j] = mj_name2id(m, mjOBJ_BODY, (std::string(feet[j]) + "_calf").c_str());
    offsets_[j] = offsets[j];
  }
  for (int j = 0; j < 12; ++j) {
    joint_adr_[j] = m->jnt_qposadr[m->actuator_trnid[2*j]];
    force_limit_[j] = std::max({std::abs(m->actuator_forcerange[2*j]),
                              std::abs(m->actuator_forcerange[2*j+1]), 1.0});
  }
  mjData* home_data = mj_makeData(m);
  mj_resetDataKeyframe(m, home_data, home_);
  mj_forward(m, home_data);
  const double* rotation = home_data->xmat + 9*torso_;
  for (int j = 0; j < 4; ++j) {
    for (int axis = 0; axis < 2; ++axis) {
      home_foot_xy_[j][axis] = 0;
      for (int k = 0; k < 3; ++k) {
        home_foot_xy_[j][axis] += rotation[3*k+axis] *
            (home_data->site_xpos[3*foot_site_[j]+k] - home_data->xpos[3*torso_+k]);
      }
    }
  }
  mj_deleteData(home_data);
}

void Task::ResidualFn::Residual(const mjModel* m, const mjData* d,
                               double* r) const {
  // Parameters: vx, vy, height, period, swing fraction, swing height, calf margin.
  const int torso = owner_->torso_;
  const double* rot = d->xmat + 9 * torso;
  const int floor = owner_->floor_;
  const double floor_z = m->geom_pos[3 * floor + 2];
  int n = 0;
  r[n++] = d->xpos[3 * torso + 2] - floor_z - parameters_[2];
  const double* vel = d->sensordata + owner_->velocity_adr_;
  r[n++] = vel[0] - parameters_[0];
  r[n++] = vel[1] - parameters_[1];
  r[n++] = rot[2]; r[n++] = rot[5]; r[n++] = rot[8] - 1;
  r[n++] = rot[0] - 1; r[n++] = rot[3];  // fixed +X heading
  for (int j = 0; j < m->nu; ++j) {
    r[n++] = d->actuator_force[j] / owner_->force_limit_[j];
  }
  const int home = owner_->home_;
  for (int j = 0; j < m->nu; ++j) {
    const int adr = owner_->joint_adr_[j];
    r[n++] = d->qpos[adr] - m->key_qpos[home*m->nq + adr];
  }
  const auto& offsets = owner_->offsets_;
  double phase = std::fmod(d->time / parameters_[3], 1.0);
  double ramp = owner_->gait_startup_s_ > 0
      ? std::clamp(d->time / owner_->gait_startup_s_, 0.0, 1.0) : 1.0;
  for (int j = 0; j < 4; ++j) {
    double p = std::fmod(phase - offsets[j] + 1.0, 1.0);
    double swing = p < parameters_[4]
        ? ramp * parameters_[5] * std::sin(3.141592653589793 * p / parameters_[4]) : 0;
    const int site = owner_->foot_site_[j];
    const int geom = owner_->foot_geom_[j];
    // Sites are sphere centers, so stance target includes the foot radius.
    r[n++] = d->site_xpos[3*site+2] - floor_z - m->geom_size[3*geom] - swing;
  }
  for (int j = 0; j < 4; ++j) {
    const int body = owner_->calf_body_[j];
    const double* zrow = d->xmat + 9*body + 6;
    const double z = d->xpos[3*body+2];
    double lower = std::min(z + .02*zrow[0] - .13*zrow[2],
                            z - .20*zrow[2]) - .01 - floor_z;
    r[n++] = std::max(parameters_[6] - lower, 0.0);
  }
  // Capture-point balance about the feet, adapted from upstream quadruped's
  // task concept. This is an independent task term, not a planner change.
  double average[2] = {0, 0};
  for (int j = 0; j < 4; ++j) {
    const double* foot = d->site_xpos + 3*owner_->foot_site_[j];
    average[0] += foot[0] / 4; average[1] += foot[1] / 4;
  }
  const double fall_time = std::sqrt(2 * parameters_[2] / 9.81);
  for (int axis = 0; axis < 2; ++axis) {
    r[n++] = d->subtree_com[3*torso+axis] + fall_time*d->qvel[axis] - average[axis];
  }
  for (int axis = 0; axis < 3; ++axis) r[n++] = d->qvel[3+axis];
  // Explicit body-frame foot placement prior, NOT an action generator. MJPC
  // still chooses all joint targets by minimizing predicted physical costs.
  for (int j = 0; j < 4; ++j) {
    const double p = std::fmod(phase - offsets[j] + 1.0, 1.0);
    const double fraction = parameters_[4];
    const double position = p < fraction
        ? -.5 * std::cos(3.141592653589793*p/fraction)
        : .5 - (p-fraction)/(1-fraction);
    for (int axis = 0; axis < 2; ++axis) {
      double relative = 0;
      for (int k = 0; k < 3; ++k) {
        relative += rot[3*k+axis] *
            (d->site_xpos[3*owner_->foot_site_[j]+k] - d->xpos[3*torso+k]);
      }
      const double stride = parameters_[axis] * parameters_[3] * (1-fraction);
      r[n++] = relative - owner_->home_foot_xy_[j][axis] - ramp*stride*position;
    }
  }
  double normal_force[4] = {0, 0, 0, 0};
  double nonfoot_penalty = 0;
  for (int i = 0; i < d->ncon; ++i) {
    const auto& contact = d->contact[i];
    if (contact.geom[0] != floor && contact.geom[1] != floor) continue;
    const int other = contact.geom[0] == floor ? contact.geom[1] : contact.geom[0];
    bool is_foot = false;
    for (int j = 0; j < 4; ++j) {
      if (other == owner_->foot_geom_[j]) {
        is_foot = true;
        double force[6];
        mj_contactForce(m, d, i, force);
        normal_force[j] += std::max(force[0], 0.0);
      }
    }
    if (!is_foot) nonfoot_penalty = std::max(nonfoot_penalty, (.002 - contact.dist) / .01);
  }
  const double body_weight = m->body_subtreemass[torso] * std::abs(m->opt.gravity[2]);
  for (int j = 0; j < 4; ++j) {
    const double p = std::fmod(phase - offsets[j] + 1.0, 1.0);
    const double load = normal_force[j] / body_weight;
    r[n++] = p < parameters_[4] ? load : std::max(.1 - load, 0.0);
  }
  r[n++] = std::max(nonfoot_penalty, 0.0);
}
}  // namespace go1
