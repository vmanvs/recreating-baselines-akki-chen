#pragma once
#include <memory>
#include <array>
#include <string>
#include <mujoco/mujoco.h>
#include "mjpc/task.h"

namespace go1 {
// Residual dimensions/order are shared with mjpc_model.py. This is our task,
// not the unavailable Go1 task from the benchmark authors.
class Task final : public mjpc::Task {
 public:
  class ResidualFn final : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const Task* task) : BaseResidualFn(task), owner_(task) {}
    void Residual(const mjModel*, const mjData*, double*) const override;
   private:
    const Task* owner_;
  };
  Task();
  std::string Name() const override { return "Go1 Matched Walk"; }
  std::string XmlPath() const override { return ""; }  // MJB supplied by Python.
 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this);
  }
  ResidualFn* InternalResidual() override { return residual_.get(); }
  void ResetLocked(const mjModel* model) override;
 private:
  // Fixed model identifiers, immutable during controller lifetime.
  int torso_, velocity_adr_, floor_, home_;
  std::array<int, 4> foot_site_, foot_geom_, calf_body_;
  std::array<int, 12> joint_adr_;
  std::array<double, 4> offsets_;
  std::array<double, 12> force_limit_;
  std::array<std::array<double, 2>, 4> home_foot_xy_;
  double gait_startup_s_;
  std::unique_ptr<ResidualFn> residual_;
};
}  // namespace go1
