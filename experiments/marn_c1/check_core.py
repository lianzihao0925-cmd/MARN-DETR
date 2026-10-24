"""CPU/CUDA numerical checks, using synthetic prototypes only; no dataset or weights."""
import io
import json
from pathlib import Path

import torch

from marn_core import BoundedMGQR, PrototypeBank, PrototypeReader, RetrievalConfig


def run(device, amp=False):
    torch.manual_seed(42)
    dim = 32
    bank = PrototypeBank([1, 3, 90], dim=dim, slots_per_class=2).to(device)
    reader = PrototypeReader(dim, [1, 3, 90], RetrievalConfig(
        key_dim=16, top_k=3, min_cosine=-1
    )).to(device)
    mgqr = BoundedMGQR(dim=dim, bottleneck=16).to(device)
    q = torch.randn(2, 7, dim, device=device)
    logits = torch.randn(2, 7, 91, device=device)
    boxes = torch.rand(2, 7, 4, device=device).clamp_min(0.01)
    selected = torch.zeros(2, 7, dtype=torch.bool, device=device)
    selected[:, [0, 3, 5]] = True

    def forward(snapshot):
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=amp):
            m, e, valid = reader(q, logits, boxes, selected, snapshot)
            return mgqr(q, m, e, valid), m, e, valid

    y, m, e, valid = forward(bank.snapshot(0))
    assert not valid.any() and torch.equal(y, q), "Empty bank must bypass exactly"
    assert torch.isfinite(m).all() and torch.isfinite(e).all()

    # Synthetic test fixture only. Production bank writes are NOT implemented in C1.
    with torch.no_grad():
        bank.features.normal_()
        bank.geometry.copy_(torch.rand_like(bank.geometry).clamp_min(0.01))
        bank.quality.fill_(0.8)
        bank.updated_step.fill_(0)
        bank.valid.fill_(True)
        # Poison an explicitly invalid slot to verify masks handle its NaN contents.
        bank.valid[-1] = False
        bank.features[-1].fill_(float("nan"))
        bank.geometry[-1].fill_(float("nan"))
        bank.quality[-1] = float("nan")

    snapshot = bank.snapshot(10)
    before = snapshot.features.clone()
    with torch.no_grad():
        bank.features[0].add_(5)
    torch.testing.assert_close(snapshot.features, before, equal_nan=True)
    y, _, _, valid = forward(snapshot)
    assert torch.equal(valid, selected), "Valid candidates expected for every selected query"
    assert torch.equal(y, q), "Zero output initialization must be exact identity"

    optim = torch.optim.SGD([*reader.parameters(), *mgqr.parameters()], lr=0.1)
    target = torch.randn_like(q)
    for step in range(2):
        optim.zero_grad(set_to_none=True)
        y, _, _, _ = forward(snapshot)
        loss = (y - target).square().mean()
        loss.backward()
        for p in [*reader.parameters(), *mgqr.parameters()]:
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), "Nonfinite gradient"
        assert mgqr.out.weight.grad.abs().sum() > 0, "Output projection must learn immediately"
        if step == 0:
            assert mgqr.fuse.weight.grad.abs().sum() == 0, "Expected zero upstream gradient at startup"
        else:
            assert mgqr.fuse.weight.grad.abs().sum() > 0, "Fusion must learn after the first update"
            assert reader.key.weight.grad is not None and reader.key.weight.grad.abs().sum() > 0
        optim.step()

    # Deliberately enlarge the output projection to exercise residual clipping.
    with torch.no_grad():
        mgqr.out.weight.normal_(std=5)
    y, _, _, _ = forward(snapshot)
    assert torch.equal(y[~selected], q[~selected]), "Unselected queries changed"
    delta = (y.float() - q.float()).norm(dim=-1)
    bound = mgqr.beta_max * mgqr.rho * q.float().norm(dim=-1)
    assert (delta <= bound + 1e-5).all(), "Residual norm bound exceeded"
    assert delta[selected].max() > 0, "A trained/nonzero module must affect selected queries"

    # A valid but oppositely directed key must be rejectable.
    reject_reader = PrototypeReader(dim, [1, 3, 90], RetrievalConfig(
        key_dim=16, top_k=2, min_cosine=0.5
    )).to(device)
    reject_bank = PrototypeBank([1, 3, 90], dim=dim, slots_per_class=1).to(device)
    with torch.no_grad():
        reject_bank.features[0].copy_(-q[0, 0])
        reject_bank.valid[0] = True
        reject_bank.quality[0] = 0.9
        reject_bank.geometry[0].fill_(0.5)
    one = torch.zeros_like(selected)
    one[0, 0] = True
    m, e, valid = reject_reader(q, logits, boxes, one, reject_bank.snapshot(0))
    assert not valid.any() and torch.equal(mgqr(q, m, e, valid), q)

    # State survives serialization; reads in eval do not mutate persistent bank data.
    bank.eval()
    states = {k: v.clone() for k, v in bank.state_dict().items()}
    forward(bank.snapshot(11))
    for k, v in bank.state_dict().items():
        torch.testing.assert_close(v, states[k], equal_nan=True)
    stream = io.BytesIO()
    torch.save(bank.state_dict(), stream)
    stream.seek(0)
    restored = PrototypeBank([1, 3, 90], dim=dim, slots_per_class=2).to(device)
    restored.load_state_dict(torch.load(stream, weights_only=True, map_location=device))
    for k, v in restored.state_dict().items():
        torch.testing.assert_close(v, states[k], equal_nan=True)

    return {"device": device, "autocast_bf16": amp, "passed": True,
            "checks": ["empty_bypass", "zero_init_identity", "snapshot_independence",
                       "invalid_nan_slot", "two_step_gradient", "unselected_identity",
                       "residual_bound", "candidate_rejection", "eval_read_only", "state_roundtrip"]}


if __name__ == "__main__":
    results = [run("cpu")]
    if torch.cuda.is_available():
        results.append(run("cuda", amp=True))
    else:
        print("CUDA_CHECK_SKIPPED: CUDA unavailable", flush=True)
    Path(__file__).with_name("core_check.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)
    print("C1_CORE_CHECK_COMPLETED", flush=True)
