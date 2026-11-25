"""Run the legacy A3 QLI pair on one device with predictable causal Top-K indices."""

import argparse

import sgl_kernel_npu  # register packaged vendors before initializing the NPU
import torch
from sgl_kernel_npu.operators.utils.cann import detect_cann_version

from sglang.srt.hardware_backend.npu.dsv4 import legacy_qli


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    print(f"CANN {detect_cann_version()}, wheel {sgl_kernel_npu.__file__}", flush=True)
    # Probe dependencies separately so the operator can build just the missing component.
    missing = []
    try:
        ops = legacy_qli._legacy_bridge()
    except (ImportError, OSError, RuntimeError) as exc:
        print(f"BRIDGE unavailable: {exc}", flush=True)
        missing.append("bridge")
    else:
        print(
            f"BRIDGE OK: {ops.npu_quant_lightning_indexer.default._schema}", flush=True
        )
    try:
        vendor = legacy_qli._legacy_vendor()
    except RuntimeError as exc:
        print(f"VENDOR unavailable: {exc}", flush=True)
        missing.append("vendor")
    else:
        print(f"VENDOR OK: {vendor}", flush=True)
    if missing:
        raise SystemExit("Unavailable component(s): " + ", ".join(missing))
    device = torch.device(f"npu:{args.device}")
    torch.npu.set_device(device)

    # Each logical key has a distinct, representable positive score. Reverse
    # the physical pages so a missing page-table lookup also fails this check.
    batch_q = [3, 5]
    pages = 20
    keys = pages * 32
    topk = 512
    raw_lengths = [keys * 4 + 1, (keys - 7) * 4 + 3]
    cu = torch.tensor([0, 3, 8], dtype=torch.int32, device=device)
    kv = torch.tensor(raw_lengths, dtype=torch.int32, device=device)
    common = dict(
        cu_seqlens_q=cu,
        seqused_k=kv // 4,
        cmp_residual_k=kv % 4,
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=3,
        cmp_ratio=4,
    )
    metadata = legacy_qli.quant_lightning_indexer_metadata(
        64, 1, 128, topk, 2, **common
    )
    q = torch.zeros(sum(batch_q), 64, 128, dtype=torch.int8)
    q[:, :, 0] = 1
    k = torch.zeros(pages, 32, 1, 128, dtype=torch.int8)
    k[:, :, :, 0] = 1
    scale = torch.arange(1, keys + 1, dtype=torch.float32).reshape(pages, 32, 1) / 1024
    table = torch.arange(pages - 1, -1, -1, dtype=torch.int32).repeat(2, 1)
    weights = torch.full((sum(batch_q), 64), 1 / 64, dtype=torch.float16, device=device)
    idx, _ = legacy_qli.quant_lightning_indexer(
        q.to(device),
        k.to(device),
        weights,
        torch.ones_like(weights),
        scale.flip(0).to(device=device, dtype=torch.float16),
        topk,
        2,
        block_table=table.to(device),
        metadata=metadata,
        return_value=0,
        **common,
    )
    expected = []
    for length, q_len in zip(raw_lengths, batch_q):
        for i in range(q_len):
            visible = (length - q_len + i + 1) // 4
            expected.append(torch.arange(visible - topk, visible, dtype=torch.int32))
    torch.testing.assert_close(
        idx.cpu().sort(dim=-1).values, torch.stack(expected)[:, None, :]
    )
    print("PASS: legacy QLI metadata and compute returned the expected Top-K on A3.")


if __name__ == "__main__":
    main()
