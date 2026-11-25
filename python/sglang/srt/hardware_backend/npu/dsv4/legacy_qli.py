"""Experimental A3 C4 QLI and its matching AICPU metadata for prefill A/B runs."""

import ctypes
import functools
import importlib
import logging
import os
from pathlib import Path

import torch
from sgl_kernel_npu.operators.indexer.indexer_utils import _cann_9_1_indexer_kwargs

logger = logging.getLogger(__name__)


def _legacy_vendor():
    # Follow the recipes bridge's search order. A system libopapi can export
    # the same compute symbol with a different ABI. Require a vendor supplying
    # BOTH the experimental compute and metadata before invoking either one.
    roots = [
        Path(p) for p in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(":") if p
    ]
    if opp := os.environ.get("ASCEND_OPP_PATH"):
        vendors = Path(opp) / "vendors"
        config = vendors / "config.ini"
        if config.is_file():
            for line in config.read_text().splitlines():
                if line.startswith("load_priority="):
                    roots.extend(
                        vendors / p.strip()
                        for p in line.partition("=")[2].split(",")
                        if p.strip()
                    )
                    break

    names = ("aclnnQuantLightningIndexer", "aclnnQuantLightningIndexerMetadata")
    for root in dict.fromkeys(roots):
        path = root / "op_api/lib/libcust_opapi.so"
        if not path.is_file():
            continue
        try:
            lib = ctypes.CDLL(str(path), mode=os.RTLD_LAZY | os.RTLD_LOCAL)
        except OSError as exc:
            logger.debug("Skipping unloadable vendor %s: %s", path, exc)
            continue
        exports = [
            hasattr(lib, name + suffix)
            for name in names
            for suffix in ("", "GetWorkspaceSize")
        ]
        if all(exports):
            return root
        if any(exports):
            raise RuntimeError(
                f"Incomplete legacy QLI pair in {path}. Put the matching "
                "experimental QLI vendor first in ASCEND_CUSTOM_OPP_PATH."
            )
    raise RuntimeError(
        "No active vendor exports the experimental QLI and QLI metadata pair. "
        "Source the QLI vendor environment before starting Python."
    )


def _legacy_bridge():
    for module, namespace in (
        ("sgl_legacy_qli", "sgl_legacy_qli"),
        ("custom_ops", "custom"),
        (
            "sgl_kernel_npu._third_party.sgl_cann_recipes_infer",
            "sgl_cann_recipes_infer",
        ),
    ):
        try:
            importlib.import_module(module)
        except ModuleNotFoundError as exc:
            if exc.name == module or module.startswith(exc.name + "."):
                continue
            raise
        ops = getattr(torch.ops, namespace)
        if all(
            hasattr(ops, name)
            for name in (
                "npu_quant_lightning_indexer",
                "npu_quant_lightning_indexer_metadata",
            )
        ):
            return ops
    raise RuntimeError(
        "No legacy QLI Torch bridge registers npu_quant_lightning_indexer and "
        "npu_quant_lightning_indexer_metadata. Install the recipes bridge."
    )


@functools.cache
def load_ops():
    ops = _legacy_bridge()
    vendor = _legacy_vendor()
    logger.info(
        "Using legacy A3 QLI %s with vendor %s", ops.npu_quant_lightning_indexer, vendor
    )
    return ops


def quant_lightning_indexer(
    q, k, weights, q_scale, k_scale, topk, quant_mode, **kwargs
):
    if quant_mode != 2:
        raise ValueError("Legacy A3 QLI requires INT8 quant_mode=2")
    return load_ops().npu_quant_lightning_indexer(
        q,
        k,
        weights,
        q_scale,
        k_scale,
        sparse_count=topk,
        **_cann_9_1_indexer_kwargs(kwargs),
    )


def quant_lightning_indexer_metadata(
    num_heads_q, num_heads_k, head_dim, topk, quant_mode, **kwargs
):
    if quant_mode != 2:
        raise ValueError("Legacy A3 QLI requires INT8 quant_mode=2")
    args = _cann_9_1_indexer_kwargs(kwargs)
    return load_ops().npu_quant_lightning_indexer_metadata(
        num_heads_q,
        num_heads_k,
        head_dim,
        sparse_count=topk,
        device=str(args["actual_seq_lengths_query"].device),
        **args,
    )
