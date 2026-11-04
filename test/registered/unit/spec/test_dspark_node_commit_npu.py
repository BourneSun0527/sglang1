"""Standalone NPU regression for sparse tree commits (no running server needed).

Run from the SGLang checkout:
    python3 test/registered/unit/spec/test_dspark_node_commit_npu.py
"""

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None

_ROOT = Path(__file__).resolve().parents[4]
_PATH = _ROOT / "python/sglang/srt/speculative/dspark_components/dspark_tree_ops.py"
_SPEC = importlib.util.spec_from_file_location("standalone_tree_ops", _PATH)
ops = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = ops
_SPEC.loader.exec_module(ops)


def reference_commit(original, nodes, table, plan, winners, active, ratio, pages):
    """CPU oracle: resolve each destination independently, snapshot all sources."""
    out = original.clone()
    source = nodes.clone()
    rows = original.shape[1]
    for page in pages:
        for row in range(rows):
            candidates = []
            for node in range(plan.n):
                req = int(plan.req_ids[node])
                winner = req * plan.max_q + int(winners[req])
                depth = int(plan.depths[node])
                if not active[req] or depth > int(plan.depths[winner]):
                    continue
                if int(plan.ancestors[winner, depth]) != node:
                    continue
                pos = int(plan.start[node]) + depth
                if (pos + 1) % ratio:
                    continue
                position = (pos + 1) // ratio - 1
                col = position // rows
                if not 0 <= col < table.shape[1]:
                    continue
                if int(table[req, col]) == page and position % rows == row:
                    candidates.append(((-req, depth, node), node))
            if candidates:
                chosen = max(candidates)[1]
                out[page, row].copy_(source[chosen])
    return out


def make_plan():
    return SimpleNamespace(
        n=10,  # non-power-of-two: exercise masked lanes in the selection kernel
        max_q=5,
        req_ids=torch.tensor([0] * 5 + [1] * 5),
        start=torch.full((10,), 126, dtype=torch.int32),
        depths=torch.tensor([0, 1, 1, 2, 3] * 2),
        ancestors=torch.tensor([
            [0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0],
            [0, 1, 3, 0], [0, 1, 3, 4],
            [5, 5, 5, 5], [5, 6, 5, 5], [5, 7, 5, 5],
            [5, 6, 8, 5], [5, 6, 8, 9],
        ]),
    )


class ReferenceTest(unittest.TestCase):
    def test_request_priority_and_latest_depth(self):
        plan = make_plan()
        original = torch.zeros(4, 2, 1)
        nodes = torch.arange(plan.n, dtype=torch.float32).reshape(plan.n, 1) + 10
        out = reference_commit(
            original, nodes, torch.ones(2, 64, dtype=torch.int32), plan,
            torch.tensor([3, 2]), torch.tensor([True, True]), 1, (1, 2),
        )
        # Depth 2 wraps onto depth 0's row; request 0 wins shared-page conflicts.
        self.assertEqual(out[1, :, 0].tolist(), [13.0, 11.0])
        self.assertTrue(torch.equal(out[2], original[2]))


@unittest.skipUnless(
    torch_npu is not None and torch.npu.is_available(), "Ascend NPU required"
)
class NodeCommitNpuTest(unittest.TestCase):
    def test_sparse_and_legacy_match_independent_reference(self):
        for ratio in (1, 4, 128):
            for active in ((True, True), (False, True), (False, False)):
                for rows, columns, padded, alias in (
                    (2, 17, False, False),
                    (4, 2051, True, False),  # byte tile tail + padded cache stride
                    (4, 2051, False, True),  # sources overlap destination pages
                ):
                    with self.subTest(ratio=ratio, active=active, rows=rows,
                                      padded=padded, alias=alias):
                        self.check_case(ratio, active, rows, columns, padded, alias)

    def check_case(self, ratio, active_flags, rows, columns, padded, alias):
        plan = make_plan()
        winners = torch.tensor([3, 2])
        active = torch.tensor(active_flags)
        table = torch.ones(2, 64, dtype=torch.int32)
        backing = torch.arange(4 * rows * (columns + 7 * padded), dtype=torch.float32)
        backing = (backing % 251).to(torch.bfloat16).reshape(4, rows, -1)
        original = backing[..., :columns]
        nodes = (
            original.reshape(-1, columns)[rows : rows + plan.n]
            if alias else torch.full((plan.n, columns), -7, dtype=torch.bfloat16)
        )
        expected = reference_commit(
            original, nodes, table, plan, winners, active, ratio, (1, 2)
        )
        # A second record exercises descriptor rows smaller than max_rows and
        # a different byte width in the same copy grid (as in state + KV).
        other = torch.full((4, 2, 9), 3, dtype=torch.float32)
        other_nodes = torch.arange(plan.n * 9, dtype=torch.float32).reshape(plan.n, 9)
        expected_other = reference_commit(
            other, other_nodes, table, plan, winners, active, ratio, (1, 2)
        )
        device_plan = SimpleNamespace(**{
            key: value.npu() if isinstance(value, torch.Tensor) else value
            for key, value in vars(plan).items()
        })
        for mode in ("legacy", "sparse"):
            device_backing = backing.npu()
            target = device_backing[..., :columns]
            source = (
                target.reshape(-1, columns)[rows : rows + plan.n]
                if alias else nodes.npu()
            )
            record = ops.NodeRecord(target, source, table.npu(), (1, 2), ratio)
            other_target = other.npu()
            other_record = ops.NodeRecord(
                other_target, other_nodes.npu(), table.npu(), (1, 2), ratio
            )
            with patch.dict(os.environ, {"SGLANG_DSPARK_NODE_COMMIT_MODE": mode}):
                prepared = ops.prepare_nodes(
                    device_plan, [record, other_record], winners.npu(), active.npu(), "npu:0"
                )
                ops.commit_nodes(prepared)
            torch.npu.synchronize()
            self.assertTrue(torch.equal(target.cpu(), expected), mode)
            self.assertTrue(torch.equal(other_target.cpu(), expected_other), mode)
            if padded:
                self.assertTrue(torch.equal(
                    device_backing[..., columns:].cpu(), backing[..., columns:]
                ), mode)

    def test_empty_records(self):
        plan = make_plan()
        for mode in ("legacy", "sparse"):
            with patch.dict(os.environ, {"SGLANG_DSPARK_NODE_COMMIT_MODE": mode}):
                # Only globalize_winners in legacy needs device plan fields;
                # max_q is a scalar and an empty endpoint tensor is sufficient.
                prepared = ops.prepare_nodes(
                    plan, [], torch.empty(0, dtype=torch.long, device="npu"),
                    torch.empty(0, dtype=torch.bool, device="npu"), "npu:0",
                )
                ops.commit_nodes(prepared)
        torch.npu.synchronize()


if __name__ == "__main__":
    unittest.main()
