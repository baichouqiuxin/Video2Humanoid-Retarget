#!/usr/bin/env bash
set -euo pipefail

ISAAC_ROOT="${ISAAC_ROOT:?Set ISAAC_ROOT to your Isaac Sim installation directory}"
USD_FILE="${1:?Usage: scripts/open_retarget_demo.sh path/to/replay.usd}"
ISAAC_APP="${ISAAC_APP:-$ISAAC_ROOT/apps/isaacsim.exp.base.kit}"

if [[ ! -f "$USD_FILE" ]]; then
  echo "USD file not found: $USD_FILE" >&2
  exit 1
fi

export PYTHONPATH=""
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_PYTHON_EXE CONDA_EXE _CONDA_EXE _CONDA_ROOT

mapfile -t CUDA_LIBS < <(find "$ISAAC_ROOT/exts/omni.isaac.ml_archive/pip_prebundle" -type d -path '*/lib' | sort)
CUDA_LIBS+=(
  "$ISAAC_ROOT/kit/python/lib/python3.11/site-packages/triton/backends/nvidia/lib/cupti"
  "$ISAAC_ROOT/extscache/omni.kit.streamsdk.plugins-7.6.3+107.0.3.lx64.r/bin"
)

LD_ADD=""
for lib_dir in "${CUDA_LIBS[@]}"; do
  if [[ -d "$lib_dir" ]]; then
    LD_ADD="${LD_ADD:+$LD_ADD:}$lib_dir"
  fi
done

export LD_LIBRARY_PATH="$LD_ADD"

TMP_SCRIPT="/tmp/open_retarget_stage.py"
cat > "$TMP_SCRIPT" <<PY
import omni.usd
import omni.kit.app
import omni.timeline
ctx = omni.usd.get_context()
ctx.open_stage("$USD_FILE")
app = omni.kit.app.get_app()
for _ in range(10):
    app.update()
timeline = omni.timeline.get_timeline_interface()
stage = ctx.get_stage()
if stage is not None:
    stage.Load()
    timeline.set_start_time(stage.GetStartTimeCode() / stage.GetTimeCodesPerSecond())
    timeline.set_end_time(stage.GetEndTimeCode() / stage.GetTimeCodesPerSecond())
    timeline.set_current_time(0.0)
    timeline.set_looping(True)
    timeline.play()
    try:
        import omni.usd
        selection = omni.usd.get_context().get_selection()
        target = "/World/G1" if stage.GetPrimAtPath("/World/G1") else "/World/G1_KinematicReplay"
        selection.set_selected_prim_paths([target], True)
    except Exception as exc:
        print(f"selection setup skipped: {exc}")
try:
    import omni.kit.viewport.utility as vp_util
    viewport = vp_util.get_active_viewport()
    if viewport is not None:
        viewport.camera_path = "/World/Camera"
        print("set viewport camera: /World/Camera")
except Exception as exc:
    print(f"viewport camera setup skipped: {exc}")
print("opened stage: $USD_FILE")
PY

exec "$ISAAC_ROOT/kit/kit" "$ISAAC_APP" --exec "$TMP_SCRIPT"
