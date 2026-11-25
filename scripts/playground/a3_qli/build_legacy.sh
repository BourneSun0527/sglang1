#!/usr/bin/env bash
# Build a missing component against the active A3 CANN 9.2 / PyTorch environment.
set -euo pipefail
case ${1:-} in
    vendor|bridge) component=$1 ;;
    *) echo "Usage: bash $0 vendor|bridge" >&2; exit 2 ;;
esac
: "${ASCEND_HOME_PATH:?Source the A3 CANN 9.2 environment before building}"
if [[ $component == bridge ]]; then
    : "${KERNEL_REPO:?Set KERNEL_REPO to the sgl-kernel-npu checkout}"
    export KERNEL_REPO
    KERNEL_REPO=$(cd "$KERNEL_REPO" && pwd)
fi

work=$(mktemp -d "${TMPDIR:-/tmp}/a3-legacy-qli.XXXXXX")
echo "Build directory: $work"
jobs=${MAX_JOBS:-8}
if [[ $component == vendor ]]; then
    url=https://gitcode.com/cann/ops-transformer.git
    rev=6f0d3ec620b338cab4f2d00322ff2e0d141419d2
else
    url=https://gitcode.com/cann/cann-recipes-infer.git
    rev=35a476c2b1ea61990380a7115ea991023aedc7d0
fi
git -C "$work" init -q
git -C "$work" remote add origin "$url"
git -C "$work" fetch --depth=1 origin "$rev"
git -C "$work" checkout -q --detach FETCH_HEAD
if [[ $component == vendor ]]; then
    git -C "$work" submodule update --init --recursive
    # The experimental build compiles opbase tiling from source. CANN 9.2
    # also ships op_common headers, with different C++ member signatures.
    cat >> "$work/CMakeLists.txt" <<'CMAKE'

if(TARGET opbase_tiling_objs)
    set(qli_opbase_include "${OPBASE_SOURCE_PATH}/pkg_inc")
    if(NOT EXISTS "${qli_opbase_include}/op_common/atvoss/elewise/elewise_tiling.h")
        message(FATAL_ERROR "Missing source-matched opbase tiling headers: ${qli_opbase_include}")
    endif()
    target_include_directories(opbase_tiling_objs BEFORE PRIVATE "${qli_opbase_include}")
endif()
CMAKE
    dest=$(realpath -m "${QLI_INSTALL_ROOT:-$PWD/qli91-opp}")
    ( cd "$work" && bash build.sh --pkg --experimental --soc=ascend910_93 \
        --vendor_name=sgl_legacy_qli --ops=quant_lightning_indexer,quant_lightning_indexer_metadata "-j$jobs" )
    shopt -s nullglob
    installers=("$work"/build_out/*.run)
    [[ ${#installers[@]} == 1 ]] || { echo "Expected one .run under $work/build_out" >&2; exit 1; }
    mkdir -p "$dest"
    bash "${installers[0]}" --quiet --install-path="$dest"
    vendor="$dest/vendors/sgl_legacy_qli"
    [[ -f "$vendor/op_api/lib/libcust_opapi.so" ]]
    printf '\nRun in the probe and P server shells before starting Python:\n'
    printf 'export ASCEND_CUSTOM_OPP_PATH=%q${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}\n' "$vendor"
else
    cd "$work"
    # Reuse the same selective recipes bridge builder as the 9.1 wheel.
    # Give the extension its own namespace so an existing HcPre can coexist.
    python3 -X utf8 - <<'PY'
import os
import sys
from pathlib import Path
import torch

kernel = Path(os.environ["KERNEL_REPO"])
sys.path[:0] = [str(kernel), str(kernel / "python/sgl_kernel_npu")]
from third_party.cann_recipes_infer.build import prepare_python

package = "sgl_legacy_qli"
prepare_python(Path.cwd(), dict(
    name=package, python_package=package,
    ops=["npu_quant_lightning_indexer", "npu_quant_lightning_indexer_metadata"],
))
header = Path("ops/ascendc/torch_ops_extension") / package / "csrc/ops_common.h"
# Torch 2.13 removed this header. Neither of the two bridges uses named tensors.
if not (Path(torch.__file__).parent / "include/ATen/NamedTensorUtils.h").is_file():
    header.write_text(header.read_text().replace("#include <ATen/NamedTensorUtils.h>", ""))
PY
    cd ops/ascendc/torch_ops_extension
    MAX_JOBS="$jobs" python3 sgl_setup.py bdist_wheel
    python3 -m pip install --no-deps --force-reinstall dist/sgl_legacy_qli-*.whl
fi
