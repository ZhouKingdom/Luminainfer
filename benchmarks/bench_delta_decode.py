"""Microbenchmark of the recurrent state step, excluding projections/conv."""
import torch
import triton
from zjhvllm.layers.gated_delta import gated_delta_decode


def main():
    torch.manual_seed(0)
    for batch in (1, 4, 16):
        h, d = 48, 128
        cache = torch.zeros(batch, 2, h, d, d, device='cuda')[:, 1]
        slots = torch.arange(batch, device='cuda')
        q, k = [torch.nn.functional.normalize(torch.randn(batch, h, d, device='cuda'), dim=-1) for _ in range(2)]
        q /= d**.5
        v = torch.randn_like(q)
        g = -torch.rand(batch, h, device='cuda')
        beta = torch.rand_like(g)

        def eager():
            state = cache[slots]
            decay = g.exp()
            delta = v - decay[..., None] * torch.einsum('bhd,bhdv->bhv', k, state)
            state = decay[..., None, None] * state + torch.einsum('bhd,bhv->bhdv', beta[..., None] * k, delta)
            out = torch.einsum('bhd,bhdv->bhv', q, state)
            cache[slots] = state
            return out

        def fused():
            return gated_delta_decode(q, k, v, g, beta, cache, slots)

        fused()
        old = triton.testing.do_bench(eager)
        new = triton.testing.do_bench(fused)
        print(f'batch={batch}: eager={old*1000:.2f} us fused={new*1000:.2f} us speedup={old/new:.2f}x')


if __name__ == '__main__':
    main()
