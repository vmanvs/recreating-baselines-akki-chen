#pragma once
#include <memory>
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
    explicit ResidualFn(const Task* task) : BaseResidualFn(task) {}
    void Residual(const mjModel*, const mjData*, double*) const override;
  };
  Task();
  std::string Name() const override { return "Go1 Matched Walk"; }
  std::string XmlPath() const override { return ""; }  // MJB supplied by Python.
 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this);
  }
  ResidualFn* InternalResidual() override { return residual_.get(); }
 private:
  std::unique_ptr<ResidualFn> residual_;
};
}  // namespace go1
