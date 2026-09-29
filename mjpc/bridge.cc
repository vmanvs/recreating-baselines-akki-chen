#include "go1_task.h"
#include <algorithm>
#include <cmath>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include "mjpc/planners/sampling/planner.h"
#include "mjpc/states/state.h"
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"

namespace {
thread_local std::string last_error;
const mjpc::ResidualFn* active_residual = nullptr;
const mjModel* active_model = nullptr;
void Sensor(const mjModel* m, mjData* d, int stage) {
  if (m == active_model && active_residual && stage == mjSTAGE_ACC) {
    // Task settings stay fixed for a rollout. Use an immutable snapshot so
    // simulation workers do not serialize on Task::Residual's mutex.
    active_residual->Residual(m, d, d->sensordata);
  }
}
struct Controller {
  mjpc::UniqueMjModel model;
  mjpc::UniqueMjData data;
  go1::Task task;
  std::unique_ptr<mjpc::ResidualFn> residual;
  mjpc::SamplingPlanner planner;
  mjpc::State state;
  mjpc::ThreadPool pool;
  int horizon, iterations;
  decltype(mjcb_sensor) previous_callback;
  Controller(const char* path, int threads, int iters)
      : model(mjpc::MakeUniqueMjModel(mj_loadModel(path, nullptr))),
        data(mjpc::MakeUniqueMjData(nullptr)), pool(threads), iterations(iters) {
    if (!model) throw std::runtime_error("Cannot load MJPC MJB model");
    if (active_residual) throw std::runtime_error("Only one MJPC controller per process");
    if (model->nq != 19 || model->nv != 18 || model->nu != 12 || model->na != 0)
      throw std::runtime_error("Expected position-actuated Go1 model dimensions");
    data = mjpc::MakeUniqueMjData(mj_makeData(model.get()));
    task.Reset(model.get());
    if (task.num_residual != 58 || task.num_term != 13 || task.parameters.size() != 7)
      throw std::runtime_error("Go1 task schema mismatch; regenerate model");
    residual = task.Residual();
    double dt = model->opt.timestep;
    horizon = std::lround(mjpc::GetNumberOrDefault(.4, model.get(), "agent_horizon") / dt) + 1;
    if (horizon < 2 || horizon > mjpc::kMaxTrajectoryHorizon)
      throw std::runtime_error("Horizon exceeds upstream planner capacity");
    const int home = mj_name2id(model.get(), mjOBJ_KEY, "home");
    mj_resetDataKeyframe(model.get(), data.get(), home);
    planner.Initialize(model.get(), task);
    planner.Allocate();
    planner.Reset(horizon, data->ctrl);
    state.Allocate(model.get());
    // Set callback only after successful construction. It never modifies the
    // Python plant model or unrelated models in this process.
    previous_callback = mjcb_sensor;
    active_model = model.get(); active_residual = residual.get(); mjcb_sensor = Sensor;
  }
  ~Controller() {
    if (active_residual == residual.get()) {
      mjcb_sensor = previous_callback;
      active_residual = nullptr; active_model = nullptr;
    }
  }
};
}  // namespace

extern "C" {
const char* go1_mjpc_error() { return last_error.c_str(); }
const char* go1_mjpc_revision() { return GO1_MJPC_REVISION; }
int go1_mjpc_mujoco_version() { return mj_version(); }
void* go1_mjpc_create(const char* model_path, int threads, int iterations) {
  try {
    if (threads < 1 || iterations < 1) throw std::runtime_error("Invalid planner counts");
    return new Controller(model_path, threads, iterations);
  } catch (const std::exception& e) { last_error = e.what(); return nullptr; }
}
void go1_mjpc_destroy(void* handle) { delete static_cast<Controller*>(handle); }
int go1_mjpc_action(void* handle, double time, const double* qpos,
                    const double* qvel, double* action, double* cost) {
  try {
    auto& c = *static_cast<Controller*>(handle);
    mj_resetData(c.model.get(), c.data.get());
    mju_copy(c.data->qpos, qpos, c.model->nq);
    mju_copy(c.data->qvel, qvel, c.model->nv);
    c.data->time = time;
    c.state.Set(c.model.get(), c.data.get());
    c.planner.SetState(c.state);
    for (int i = 0; i < c.iterations; ++i) c.planner.OptimizePolicy(c.horizon, c.pool);
    c.planner.ActionFromPolicy(action, c.state.state().data(), time);
    const auto* best = c.planner.BestTrajectory();
    if (!best || best->failure) throw std::runtime_error("All selected MJPC rollouts diverged");
    *cost = best->total_return;
    return 0;
  } catch (const std::exception& e) { last_error = e.what(); return -1; }
}
int go1_mjpc_residual(void* handle, double time, const double* qpos,
                      const double* qvel, const double* ctrl, double* residual) {
  try {
    auto& c = *static_cast<Controller*>(handle);
    mj_resetData(c.model.get(), c.data.get());
    mju_copy(c.data->qpos, qpos, c.model->nq);
    mju_copy(c.data->qvel, qvel, c.model->nv);
    mju_copy(c.data->ctrl, ctrl, c.model->nu);
    c.data->time = time;
    mj_forward(c.model.get(), c.data.get());
    c.task.Residual(c.model.get(), c.data.get(), residual);
    return 0;
  } catch (const std::exception& e) { last_error = e.what(); return -1; }
}
}
