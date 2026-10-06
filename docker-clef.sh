#!/usr/bin/env bash
# Clef-Flash server in Docker on DGX Spark / GB10.   usage: ./docker-clef.sh [build|start|stop|restart|status|logs|ensure|install-boot]
#   HOST   name or address to publish on. Required for the first `start`; there is no default. 127.0.0.1 = this machine only, 0.0.0.0 = every
#          interface, or the name/address of one interface (e.g. a VPN address, to serve that network only).
#   PORT   published port (default 8100)
#   Without HOST / PORT, every command uses the address the existing container is already published on.
#   CLEF_QUANT / CLEF_ACT_SCALE / CLEF_GPU_FRACTION / CLEF_MAX_BATCH / CLEF_MAX_PAD_TOKENS   passed through to server.py
# Restarts: the container runs with --restart unless-stopped (comes back after a crash or a reboot; `stop` keeps it down, also across reboots).
#   At boot Docker can start before the published address is up (a VPN interface) or before the GPU's CDI spec exists; such a failed start is NOT retried by
#   Docker. `install-boot` adds a systemd user unit that runs `ensure` after boot: wait for the address and Docker, then start the container.
# No authentication: anyone who can reach HOST:PORT can use it.
# `build` needs only uv, Docker and the CUDA toolkit: it creates .venv/ from uv.lock, quantizes the model into clef-flash-nvfp4/ when that is
#   missing, and bakes both into the image (no mounts, ~12 GB of memory).
# GPU access is CDI-only on DGX OS (--device nvidia.com/gpu=all; --gpus all does not work).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
IMAGE="${IMAGE:-clef-flash-server:local}"; CONTAINER="${CONTAINER:-clef-flash}"
HOST="${HOST:-}"; PORT="${PORT:-}"; IP=""; READY_TIMEOUT="${READY_TIMEOUT:-900}"
STOPPED="${HERE}/.clef-docker.stopped"      # written by `stop`: tells `ensure` the container is down on purpose
UNIT="clef-flash-boot.service"
CAUSAL_CONV1D_VERSION="${CAUSAL_CONV1D_VERSION:-1.7.0}"

published() {   # "ip port" the existing container is published on; empty when there is no container
  docker inspect -f '{{range .HostConfig.PortBindings}}{{range .}}{{.HostIp}} {{.HostPort}}{{end}}{{end}}' "${CONTAINER}" 2>/dev/null || true
}
resolve() {     # sets IP, HOST and PORT: from the HOST / PORT given, otherwise from the existing container
  local cur; cur="$(published)"
  [[ -n "${PORT}" ]] || PORT="${cur#* }"; [[ -n "${PORT}" ]] || PORT=8100
  if [[ -z "${HOST}" ]]; then
    IP="${cur% *}"; HOST="${IP}"
    [[ -n "${IP}" ]] || { echo "ERROR: set HOST to the address to publish on: HOST=127.0.0.1 (this machine only), HOST=0.0.0.0 (every interface), or the name/address of one interface" >&2; exit 1; }
  elif [[ "${HOST}" =~ ^[0-9.]+$ ]]; then IP="${HOST}"
  else
    IP="$(getent hosts "${HOST}" | awk '{print $1; exit}')"
    [[ -n "${IP}" ]] || IP="$(tailscale status 2>/dev/null | awk -v h="${HOST}" '$2 == h {print $1; exit}')"
    [[ -n "${IP}" ]] || { echo "ERROR: cannot resolve HOST=${HOST}" >&2; exit 1; }
  fi
}
do_build() {
  command -v uv >/dev/null || { echo "ERROR: uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }
  local site=.venv/lib/python3.13/site-packages
  if [[ -z "$(ls ${site}/causal_conv1d_cuda.*.so 2>/dev/null)" ]]; then
    command -v nvcc >/dev/null || { echo "ERROR: the CUDA toolkit is required to compile causal-conv1d (nvcc is not on PATH)" >&2; exit 1; }
    echo "first sync: compiling causal-conv1d takes about 6 minutes"
  fi
  uv sync --locked                                   # .venv/ from uv.lock; the image installs the same lock
  if [[ ! -f clef-flash-nvfp4/model_nvfp4.safetensors ]]; then
    avail=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
    (( avail >= 30 )) || { echo "ERROR: only ${avail} GB MemAvailable, the export needs ~30 GB" >&2; exit 1; }
    echo "clef-flash-nvfp4/ is missing: exporting it (downloads the 19 GB bf16 release to clef-flash/ unless it is already there)"
    .venv/bin/python export_nvfp4.py
  fi
  # causal-conv1d cannot be compiled in the image (no CUDA toolkit there): hand over the build from .venv/, made against the same torch
  rm -rf docker/prebuilt && mkdir -p docker/prebuilt
  cp -r ${site}/causal_conv1d ${site}/causal_conv1d-*.dist-info ${site}/causal_conv1d_cuda.*.so docker/prebuilt/
  docker build -f docker/Dockerfile -t "${IMAGE}" .
}
do_start() {
  docker image inspect "${IMAGE}" >/dev/null 2>&1 || do_build
  resolve; local ip="${IP}"
  avail=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
  (( avail >= 14 )) || { echo "ERROR: only ${avail} GB MemAvailable, need ~14 GB" >&2; exit 1; }
  docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true; rm -f "${STOPPED}"
  docker run -d --name "${CONTAINER}" \
    --device nvidia.com/gpu=all \
    --restart unless-stopped \
    -p "${ip}:${PORT}:8100" \
    -v clef-flash-cache:/cache \
    -e CLEF_QUANT="${CLEF_QUANT:-nvfp4}" -e CLEF_ACT_SCALE="${CLEF_ACT_SCALE:-static}" -e CLEF_GPU_FRACTION="${CLEF_GPU_FRACTION:-0.14}" \
    -e CLEF_MAX_BATCH="${CLEF_MAX_BATCH:-8}" -e CLEF_MAX_PAD_TOKENS="${CLEF_MAX_PAD_TOKENS:-8192}" \
    "${IMAGE}" >/dev/null
  deadline=$(( SECONDS + READY_TIMEOUT ))
  until curl -sf -m 3 "http://${ip}:${PORT}/health" 2>/dev/null | grep -q '"ok":true'; do
    docker ps -q -f "name=^${CONTAINER}$" | grep -q . || { echo "container exited:"; docker logs --tail 40 "${CONTAINER}"; exit 1; }
    (( SECONDS < deadline )) || { echo "not ready after ${READY_TIMEOUT}s"; docker logs --tail 20 "${CONTAINER}"; exit 1; }
    sleep 3
  done
  echo "READY on http://${HOST}:${PORT} (${ip})  $(docker logs "${CONTAINER}" 2>&1 | grep 'warm and ready' | tail -1)"
}
do_stop() { touch "${STOPPED}"; docker stop -t 30 "${CONTAINER}" >/dev/null 2>&1 && echo "stopped (logs kept: docker logs ${CONTAINER}); stays down until \`start\`" || echo "not running"; }
do_ensure() {
  # Boot-time safety net. Does nothing if the container was stopped on purpose or does not exist.
  [[ -f "${STOPPED}" ]] && { echo "stopped on purpose; not starting"; return 0; }
  local deadline=$(( SECONDS + ${ENSURE_TIMEOUT:-600} )) what="docker"
  until docker info >/dev/null 2>&1 && what="the GPU CDI spec" && [[ -e /var/run/cdi/nvidia.yaml || -n "$(ls /etc/cdi 2>/dev/null)" ]] \
        && { docker inspect "${CONTAINER}" >/dev/null 2>&1 || { echo "no container ${CONTAINER}; run: HOST=... $0 start" >&2; exit 1; }; } \
        && resolve && what="the address ${IP}" && { [[ "${IP}" == "0.0.0.0" ]] || ip -4 -o addr | grep -qw "${IP}"; }; do
    (( SECONDS < deadline )) || { echo "${what} not ready after ${ENSURE_TIMEOUT:-600}s" >&2; exit 1; }
    sleep 5
  done
  for _ in 1 2 3 4 5 6; do
    [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER}")" == "true" ]] && { echo "running"; return 0; }
    docker start "${CONTAINER}" >/dev/null 2>&1 || true; sleep 10
  done
  echo "could not start ${CONTAINER}: $(docker inspect -f '{{.State.Error}}' "${CONTAINER}")" >&2; exit 1
}
do_install_boot() {
  mkdir -p "${HOME}/.config/systemd/user"
  cat > "${HOME}/.config/systemd/user/${UNIT}" <<UNITEOF
[Unit]
Description=Start the clef-flash container once Docker, its published address and the GPU are up

[Service]
Type=oneshot
Environment=CONTAINER=${CONTAINER}
ExecStart=${HERE}/docker-clef.sh ensure

[Install]
WantedBy=default.target
UNITEOF
  systemctl --user daemon-reload && systemctl --user enable "${UNIT}"
  if [[ "$(loginctl show-user "${USER}" -p Linger --value 2>/dev/null)" != "yes" ]]; then
    loginctl enable-linger "${USER}" 2>/dev/null || echo "NOTE: user services only start at boot with lingering on. Run once:  sudo loginctl enable-linger ${USER}"
  fi
  echo "installed ${UNIT} (linger: $(loginctl show-user "${USER}" -p Linger --value 2>/dev/null))"
}
do_status() {
  docker ps -a -f "name=^${CONTAINER}$" --format 'container: {{.Names}}  {{.Status}}  {{.Ports}}'
  resolve; curl -s -m 3 "http://${IP}:${PORT}/health" || echo "(no answer)"; echo
  awk '/MemAvailable/ {printf "MemAvailable: %.1f GB\n", $2/1048576}' /proc/meminfo
}
case "${1:-start}" in
  build) do_build ;; start) do_start ;; stop) do_stop ;; restart) do_stop; do_start ;; status) do_status ;; logs) docker logs -f "${CONTAINER}" ;;
  ensure) do_ensure ;; install-boot) do_install_boot ;;
  *) echo "usage: $0 [build|start|stop|restart|status|logs|ensure|install-boot]" >&2; exit 2 ;;
esac
