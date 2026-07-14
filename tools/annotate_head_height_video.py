#!/usr/bin/env python3
"""Overlay head-height readouts onto an existing video."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def load_heights(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    heights = np.asarray(data["height_m"], dtype=np.float64)
    deltas = np.asarray(data.get("delta_height_m", heights - heights[0]), dtype=np.float64)
    if heights.size == 0:
        raise ValueError(f"No height samples in {path}")
    return heights, deltas


def draw_panel(frame: np.ndarray, lines: list[str]) -> None:
    height, width = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.8, width / 1920.0)
    thickness = max(2, int(round(width / 960.0)))
    line_height = int(round(34 * scale))
    pad_x = int(round(18 * scale))
    pad_y = int(round(14 * scale))
    text_width = 0
    for line in lines:
        (size, _) = cv2.getTextSize(line, font, scale, thickness)
        text_width = max(text_width, size[0])
    panel_width = text_width + 2 * pad_x
    panel_height = len(lines) * line_height + 2 * pad_y
    x1 = width - panel_width - int(round(28 * scale))
    y1 = int(round(26 * scale))
    x2 = width - int(round(28 * scale))
    y2 = y1 + panel_height

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (28, 28, 28), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0.0, dst=frame)
    for index, line in enumerate(lines):
        y = y1 + pad_y + int(round((index + 0.78) * line_height))
        cv2.putText(frame, line, (x1 + pad_x, y), font, scale, (245, 245, 245), thickness, cv2.LINE_AA)


def annotate_video(input_video: Path, height_json: Path, output_video: Path, label: str, crf: int) -> None:
    heights, deltas = load_heights(height_json)
    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {input_video}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.8f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_video),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None

    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            sample_index = min(frame_index, len(heights) - 1)
            draw_panel(
                frame,
                [
                    label,
                    f"Head relZ: {deltas[sample_index]:+.2f} m",
                    f"Head Z: {heights[sample_index]:.2f} m",
                ],
            )
            process.stdin.write(frame.tobytes())
            frame_index += 1
    finally:
        capture.release()
        process.stdin.close()

    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}")
    print(f"wrote {output_video} frames={frame_index} fps={fps:.3f} size={width}x{height}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--height-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="Isaac Sim")
    parser.add_argument("--crf", type=int, default=18)
    args = parser.parse_args()
    annotate_video(args.input, args.height_json, args.output, args.label, args.crf)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
