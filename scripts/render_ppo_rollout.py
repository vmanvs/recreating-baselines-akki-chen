"""Render a saved Go1 evaluation trajectory using its recorded MuJoCo states.

Run inside the Linux RL environment with imageio-ffmpeg installed:
  python scripts/render_ppo_rollout.py --run ppo-seed-000 \
    --evaluation outputs/verify-ppo-seed-000 --output outputs/verify-ppo.mp4
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import imageio_ffmpeg
import mediapy as media
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from go1_benchmark.fixed_velocity_env import make_fixed_velocity_env


def render(run: Path, evaluation: Path, output: Path) -> dict:
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    trajectory_path = evaluation / "trajectory.npz"
    with np.load(trajectory_path) as saved:
        qpos = saved["qpos"]
        qvel = saved["qvel"]
        time_s = saved["time_s"]
        forward_velocity = saved["local_linvel"][:, 0]
        contact_flags = saved["nonfoot_ground_contact"].astype(bool)

    dt = float(manifest["environment"]["control_dt_s"])
    if len(time_s) != len(qpos) or not np.allclose(np.diff(time_s), dt, atol=1e-3):
        raise ValueError("Trajectory timing does not match the run's control rate.")

    env = make_fixed_velocity_env(
        command=tuple(manifest["command_m_s_rad_s"]),
        impl=manifest["environment"]["implementation"],
        full_collisions=True,
        noise_level=0.0,
        randomize_initial_state=False,
        settings=manifest["environment"],
    )
    model = env.mj_model
    data = mujoco.MjData(model)
    if qpos.shape[1] != model.nq or qvel.shape[1] != model.nv:
        raise ValueError("Trajectory state dimensions do not match the MuJoCo model.")

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = 135
    camera.elevation = -17
    camera.distance = 1.7
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 22)
    floor_id = model.geom("floor").id
    feet = {model.geom(name).id for name in ("FR", "FL", "RR", "RL")}
    contact_geoms: collections.Counter[str] = collections.Counter()
    penetrations_mm: dict[str, list[float]] = collections.defaultdict(list)
    flagged_with_cpu_pair = 0
    width, height = 960, 540
    output.parent.mkdir(parents=True, exist_ok=True)
    media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())

    with mujoco.Renderer(model, width=width, height=height) as renderer:  # noqa: SIM117
        with media.VideoWriter(output, (height, width), fps=round(1 / dt)) as writer:
            for i in range(len(time_s)):
                data.qpos[:] = qpos[i]
                data.qvel[:] = qvel[i]
                data.time = float(time_s[i])
                mujoco.mj_forward(model, data)

                if contact_flags[i]:
                    found_pair = False
                    for j in range(data.ncon):
                        contact = data.contact[j]
                        if contact.dist > 0:
                            continue
                        a, b = int(contact.geom1), int(contact.geom2)
                        if floor_id not in (a, b):
                            continue
                        other = b if a == floor_id else a
                        if other in feet:
                            continue
                        name = model.geom(other).name
                        contact_geoms[name] += 1
                        penetrations_mm[name].append(-1000.0 * float(contact.dist))
                        found_pair = True
                    flagged_with_cpu_pair += int(found_pair)

                camera.lookat[:] = (float(qpos[i, 0]) + 0.1, float(qpos[i, 1]), 0.22)
                renderer.update_scene(data, camera=camera)
                frame = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, width, 60), fill=(18, 22, 28))
                draw.text(
                    (16, 12),
                    f"{time_s[i]:5.2f} s   forward {forward_velocity[i]:+.2f} m/s"
                    f"   target {manifest['command_m_s_rad_s'][0]:+.2f} m/s",
                    fill="white",
                    font=font,
                )
                if contact_flags[i]:
                    draw.rectangle((width - 300, 60, width, 103), fill=(140, 22, 22))
                    draw.text(
                        (width - 286, 67),
                        "NON-FOOT CONTACT",
                        fill="white",
                        font=font,
                    )
                writer.add_image(np.asarray(frame))

    report = {
        "source_trajectory": str(trajectory_path),
        "source_checkpoint": json.loads(
            (evaluation / "summary.json").read_text(encoding="utf-8")
        )["checkpoint"],
        "frames": len(time_s),
        "fps": round(1 / dt),
        "recorded_contact_frames": int(contact_flags.sum()),
        "cpu_reconstruction_matched_frames": flagged_with_cpu_pair,
        "cpu_reconstructed_contact_geometries": dict(contact_geoms),
        "cpu_reconstructed_penetration_mm": {
            name: {
                "median": round(float(np.median(depths)), 3),
                "max": round(max(depths), 3),
            }
            for name, depths in penetrations_mm.items()
        },
        "note": (
            "Video replays recorded MJX poses through MuJoCo's renderer. "
            "Geometry names come from CPU contact reconstruction and may differ "
            "from the original MJX collision calculation."
        ),
    }
    (output.parent / f"{output.stem}-render-analysis.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.run, args.evaluation, args.output), indent=2))


if __name__ == "__main__":
    main()
