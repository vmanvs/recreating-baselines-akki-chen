#pragma once
#include <mujoco/mujoco.h>

// Adapter for the pinned MJPC source against the project's MuJoCo 3.12 wheel.
// No change to sampling, costs or dynamics. These legacy error helpers now
// share the variadic mju_error interface; mj_ray gained a normal output.
#if mjVERSION_HEADER >= 3012000
#define mju_error_s mju_error
#define mju_error_i mju_error
inline mjtNum go1_legacy_mj_ray(const mjModel* model, const mjData* data,
                              const mjtNum* point, const mjtNum* direction,
                              const mjtByte* group, mjtBool static_geom,
                              int excluded_body, int* geom_id) {
  return mj_ray(model, data, point, direction, group, static_geom,
                excluded_body, geom_id, nullptr);
}
#define mj_ray go1_legacy_mj_ray
#endif
