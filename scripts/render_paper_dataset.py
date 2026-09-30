"""Render prespecified seed-100 comparisons from saved poses only."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio_ffmpeg
import mediapy as media
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

LABELS = {
    "ppo_reference": "PPO reference",
    "ppo_gait": "PPO gait",
    "ppo_calf": "PPO gait + calf",
    "mjpc": "MJPC",
}


def render_group(dataset, identifiers, output, duration_s, preview_s):
    runs = []
    for identifier in identifiers:
        run = dataset / "runs" / identifier
        summary = json.loads((run / "summary.json").read_text())
        with np.load(run / "trajectory.npz") as saved:
            trajectory = {key: saved[key].copy() for key in saved.files}
        runs.append((summary, trajectory))
    model = mujoco.MjModel.from_binary_path(
        str(dataset / "runs" / identifiers[0] / "model.mjb")
    )
    width, height = 640, 400
    model.vis.global_.offwidth = max(width, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(height, model.vis.global_.offheight)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth, camera.elevation, camera.distance = 135, -20, 1.65
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    columns, rows = 2, (len(runs) + 1) // 2
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    preview_written = False
    stopped_frames = {}
    with (
        mujoco.Renderer(model, width=width, height=height) as renderer,
        media.VideoWriter(output, (height * rows, width * columns), fps=25) as writer,
    ):
        for frame_index in range(round(duration_s * 25)):
            replay_time = 0.02 + frame_index / 25
            canvas = Image.new("RGB", (width * columns, height * rows), (18, 22, 28))
            for panel, (summary, saved) in enumerate(runs):
                index = max(
                    0,
                    min(
                        np.searchsorted(saved["time_s"], replay_time, side="right") - 1,
                        len(saved["time_s"]) - 1,
                    ),
                )
                stopped = (
                    summary["first_failure_time_s"] is not None
                    and replay_time >= summary["first_failure_time_s"]
                )
                recorded_time = float(saved["time_s"][index])
                if panel in stopped_frames:
                    frame = stopped_frames[panel].copy()
                else:
                    data.qpos[:] = saved["qpos"][index]
                    data.qvel[:] = saved["qvel"][index]
                    data.ctrl[:] = saved["action"][index]
                    data.time = recorded_time
                    mujoco.mj_forward(model, data)
                    camera.lookat[:] = [data.qpos[0] + 0.1, data.qpos[1], 0.22]
                    renderer.update_scene(data, camera=camera)
                    # Appearance only; recorded motion is never changed.
                    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
                    frame = Image.fromarray(renderer.render())
                    if stopped:
                        stopped_frames[panel] = frame.copy()
                draw = ImageDraw.Draw(frame)
                draw.rectangle(
                    (0, 0, width, 76), fill=(70, 18, 18) if stopped else (18, 22, 28)
                )
                draw.text(
                    (10, 5),
                    f"{LABELS[summary['controller']]} | replay {replay_time:.2f}s",
                    fill="white",
                    font=font,
                )
                draw.text(
                    (10, 28),
                    f"recorded {recorded_time:.3f}s | vx {saved['local_linvel'][index, 0]:+.2f}/0.50 m/s",
                    fill="white",
                    font=font,
                )
                if stopped:
                    message = f"STOPPED: {summary['failure_reason']}"
                elif summary["force_n"] and 5 <= replay_time < 5.2:
                    message = f"PUSH: {summary['force_n']} N, world {summary['direction_deg']} degrees"
                else:
                    message = f"decision {saved['planning_time_s'][index] * 1000:.1f} ms | offline replay"
                draw.text((10, 51), message, fill="white", font=font)
                canvas.paste(
                    frame, ((panel % columns) * width, (panel // columns) * height)
                )
            if not preview_written and replay_time >= preview_s:
                canvas.save(output.with_suffix(".png"))
                preview_written = True
            writer.add_image(np.asarray(canvas))
            if frame_index % 125 == 0:
                print(
                    f"{output.name}: replay {replay_time:.2f}/{duration_s:g}s",
                    flush=True,
                )
    print(output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render_group(
        args.dataset,
        [f"{c}-nominal-100" for c in LABELS],
        args.output / "nominal-seed-100.mp4",
        30,
        4,
    )
    render_group(
        args.dataset,
        [f"{c}-push-50-90-100" for c in ("ppo_calf", "mjpc")],
        args.output / "push-50N-plusY-seed-100.mp4",
        12,
        5.16,
    )


if __name__ == "__main__":
    main()
