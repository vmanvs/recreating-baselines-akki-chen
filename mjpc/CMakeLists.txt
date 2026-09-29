cmake_minimum_required(VERSION 3.24)
project(go1_mjpc LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include(FetchContent)
find_package(Threads REQUIRED)

# Compile upstream's actual planner, not a local reimplementation. Avoid the
# GUI/gRPC and their unrelated robot asset downloads. Pin both dependencies.
set(GO1_MJPC_REVISION ff572a21e7c2bf9fda62e1862a758da7e9a8719b)
FetchContent_Declare(mjpc_source
  GIT_REPOSITORY https://github.com/google-deepmind/mujoco_mpc.git
  GIT_TAG ${GO1_MJPC_REVISION})
FetchContent_GetProperties(mjpc_source)
if(NOT mjpc_source_POPULATED)
  # Do not invoke upstream's top-level GUI build.
  FetchContent_Populate(mjpc_source)
endif()
set(ABSL_PROPAGATE_CXX_STD ON)
set(ABSL_BUILD_TESTING OFF)
FetchContent_Declare(absl
  GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
  GIT_TAG fb3621f4f897824c0dbe0615fa94543df6192f30)
FetchContent_MakeAvailable(absl)

# Use headers and library from the SAME Python MuJoCo installation as the
# evaluator. Do not compile against one MuJoCo version and load another.
if(NOT EXISTS "${MUJOCO_INCLUDE_DIR}/mujoco/mujoco.h" OR
   NOT EXISTS "${MUJOCO_LIBRARY}")
  message(FATAL_ERROR "Set MUJOCO_INCLUDE_DIR and MUJOCO_LIBRARY via scripts/build_mjpc.py")
endif()
add_library(mujoco_wheel SHARED IMPORTED)
set_target_properties(mujoco_wheel PROPERTIES
  IMPORTED_LOCATION "${MUJOCO_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${MUJOCO_INCLUDE_DIR}")

set(core_sources
  mjpc/task.cc mjpc/norm.cc mjpc/utilities.cc mjpc/trajectory.cc
  mjpc/threadpool.cc mjpc/states/state.cc mjpc/spline/spline.cc
  mjpc/planners/planner.cc mjpc/planners/sampling/planner.cc
  mjpc/planners/sampling/policy.cc)
list(TRANSFORM core_sources PREPEND "${mjpc_source_SOURCE_DIR}/")
add_library(go1_mjpc SHARED bridge.cc go1_task.cc ${core_sources})
target_compile_options(go1_mjpc PRIVATE
  -include "${CMAKE_CURRENT_SOURCE_DIR}/mujoco_compat.h")
target_include_directories(go1_mjpc PRIVATE ${mjpc_source_SOURCE_DIR})
target_link_libraries(go1_mjpc PRIVATE mujoco_wheel Threads::Threads
  absl::flat_hash_map absl::check absl::strings absl::random_random absl::span)
target_compile_definitions(go1_mjpc PRIVATE
  GO1_MJPC_REVISION="${GO1_MJPC_REVISION}"
  SOURCE_DIR="${mjpc_source_SOURCE_DIR}")
get_filename_component(mujoco_libdir "${MUJOCO_LIBRARY}" DIRECTORY)
set_target_properties(go1_mjpc PROPERTIES BUILD_RPATH "${mujoco_libdir}")
