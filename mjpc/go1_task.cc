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

void Task::ResidualFn::Residual(const mjModel* m, const mjData* d,
                               double* r) const {
  // Parameters: vx, vy, height, period, swing fraction, swing height, calf margin.
  const int torso = mj_name2id(m, mjOBJ_BODY, "trunk");
  const double* rot = d->xmat + 9 * torso;
  const int floor = mj_name2id(m, mjOBJ_GEOM, "floor");
  const double floor_z = m->geom_pos[3 * floor + 2];
  int n = 0;
  r[n++] = d->xpos[3 * torso + 2] - floor_z - parameters_[2];
  const double* vel = mjpc::SensorByName(m, d, "local_linvel");
  r[n++] = vel[0] - parameters_[0];
  r[n++] = vel[1] - parameters_[1];
  r[n++] = rot[2]; r[n++] = rot[5]; r[n++] = rot[8] - 1;
  r[n++] = rot[0] - 1; r[n++] = rot[3];  // fixed +X heading
  for (int j = 0; j < m->nu; ++j) {
    const double limit = std::max(std::abs(m->actuator_forcerange[2*j]),
                                  std::abs(m->actuator_forcerange[2*j+1]));
    r[n++] = d->actuator_force[j] / std::max(limit, 1.0);
  }
  const int home = mj_name2id(m, mjOBJ_KEY, "home");
  for (int j = 0; j < m->nu; ++j) {
    const int joint = m->actuator_trnid[2*j];
    const int adr = m->jnt_qposadr[joint];
    r[n++] = d->qpos[adr] - m->key_qpos[home*m->nq + adr];
  }
  const char* feet[] = {"FR", "FL", "RR", "RL"};
  const double* offsets = mjpc::GetCustomNumericData(m, "go1_gait_offsets");
  double phase = std::fmod(d->time / parameters_[3], 1.0);
  for (int j = 0; j < 4; ++j) {
    double p = std::fmod(phase - offsets[j] + 1.0, 1.0);
    double swing = p < parameters_[4]
        ? parameters_[5] * std::sin(3.141592653589793 * p / parameters_[4]) : 0;
    const int site = mj_name2id(m, mjOBJ_SITE, feet[j]);
    const int geom = mj_name2id(m, mjOBJ_GEOM, feet[j]);
    // Sites are sphere centers, so stance target includes the foot radius.
    r[n++] = d->site_xpos[3*site+2] - floor_z - m->geom_size[3*geom] - swing;
  }
  for (int j = 0; j < 4; ++j) {
    const std::string calf = std::string(feet[j]) + "_calf";
    const int body = mj_name2id(m, mjOBJ_BODY, calf.c_str());
    const double* zrow = d->xmat + 9*body + 6;
    const double z = d->xpos[3*body+2];
    double lower = std::min(z + .02*zrow[0] - .13*zrow[2],
                            z - .20*zrow[2]) - .01 - floor_z;
    r[n++] = std::max(parameters_[6] - lower, 0.0);
  }
}
}  // namespace go1
