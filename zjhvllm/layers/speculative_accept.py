"""Per-request greedy prefix acceptance; no cross-request length reduction."""
import torch
import triton
import triton.language as tl


@triton.jit
def _accept(Target, Inputs, Cu, Budgets, IgnoreEOS, Output,
            EOS: tl.constexpr, WIDTH: tl.constexpr):
    request = tl.program_id(0)
    start = tl.load(Cu + request)
    end = tl.load(Cu + request + 1)
    size = end - start
    budget = tl.load(Budgets + request)
    ignore = tl.load(IgnoreEOS + request)
    done = False
    emitted = 0
    accepted = 0
    for pos in range(WIDTH):
        if not done and pos < size and pos < budget:
            token = tl.load(Target + start + pos)
            matches = False
            if pos + 1 < size:
                matches = token == tl.load(Inputs + start + pos + 1)
            tl.store(Output + request * (WIDTH + 2) + pos, token)
            emitted += 1
            accepted += matches.to(tl.int32)
            done = (matches == 0) | ((ignore == 0) & (token == EOS))
    tl.store(Output + request * (WIDTH + 2) + WIDTH, emitted)
    tl.store(Output + request * (WIDTH + 2) + WIDTH + 1, accepted)


def greedy_accept_batch(target, inputs, cu_seqlens, budgets, ignore_eos, eos, width):
    """Inputs contain [pending, draft...] for each request; target has one
    prediction per input row. Return variable-length outputs and draft counts.
    The final two columns share the one small device-to-host transfer.
    """
    batch = len(budgets)
    if not 1 <= width <= 33 or batch != len(ignore_eos):
        raise ValueError("invalid acceptance dimensions")
    if any(budget < 1 for budget in budgets):
        raise ValueError("positive per-request output budgets required")
    packed = torch.full((batch, width + 2), -1, device=target.device, dtype=torch.int64)
    _accept[(batch,)](
        target, inputs, cu_seqlens,
        torch.tensor(budgets, device=target.device, dtype=torch.int32),
        torch.tensor(ignore_eos, device=target.device, dtype=torch.int32),
        packed, EOS=eos, WIDTH=width, num_warps=1)
    rows = packed.tolist()
    return [row[:row[width]] for row in rows], [row[width + 1] for row in rows]
