"""Replay an MJPC trajectory, without rerunning planning or changing its poses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def render(evaluation: Path, output: Path, preview_time_s: float = 4):
    import imageio_ffmpeg
    import mediapy as media
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    config = json.loads((evaluation / "experiment.json").read_text())
    model = mujoco.MjModel.from_binary_path(str(evaluation / "model.mjb"))
    data = mujoco.MjData(model)
    dt = config["environment"]["control_dt_s"]
    command = config["environment"]["command_m_s_rad_s"][0]
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth, camera.elevation, camera.distance = 135, -20, 1.7
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    width, height = 960, 540
    model.vis.global_.offwidth = max(width, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(height, model.vis.global_.offheight)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    preview_saved = False
    with np.load(evaluation / "trajectory.npz") as saved:
        if not np.allclose(np.diff(saved["time_s"]), dt):
            raise ValueError("Unexpected trajectory timing")
        with mujoco.Renderer(model, width=width, height=height) as renderer:  # noqa: SIM117
            with media.VideoWriter(
                output, (height, width), fps=round(1 / dt)
            ) as writer:
                for i, time_s in enumerate(saved["time_s"]):
                    data.qpos[:] = saved["qpos"][i]
                    data.qvel[:] = saved["qvel"][i]
                    data.ctrl[:] = saved["action"][i]
                    data.time = float(time_s)
                    mujoco.mj_forward(model, data)
                    camera.lookat[:] = (data.qpos[0] + 0.1, data.qpos[1], 0.22)
                    renderer.update_scene(data, camera=camera)
                    frame = Image.fromarray(renderer.render())
                    draw = ImageDraw.Draw(frame)
                    draw.rectangle((0, 0, width, 66), fill=(18, 22, 28))
                    label = (
                        f"MJPC  sim {time_s:.2f}s  "
                        f"vx {saved['local_linvel'][i, 0]:+.2f}/{command:.2f} m/s  "
                        f"plan {saved['planning_time_s'][i] * 1000:.0f} ms"
                    )
                    draw.text((14, 7), label, fill="white", font=font)
                    draw.text(
                        (14, 35),
                        "Offline simulation replay, not real-time execution",
                        fill="white",
                        font=font,
                    )
                    if saved["nonfoot_ground_contact_any_substep"][i]:
                        draw.text((14, 76), "NON-FOOT CONTACT", fill="red", font=font)
                    if not preview_saved and time_s >= preview_time_s:
                        frame.save(output.with_suffix(".png"))
                        preview_saved = True
                    writer.add_image(np.asarray(frame))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview-time-s", type=float, default=4)
    args = parser.parse_args()
    print(render(args.evaluation, args.output, args.preview_time_s))


if __name__ == "__main__":
    main()
