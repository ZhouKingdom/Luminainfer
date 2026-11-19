"""Small-batch invariance is a stronger assertion than numerical closeness."""
import pytest
import torch
from zjhvllm.layers.invariant_linear import invariant_linear


@pytest.fixture(scope="session", autouse=True)
def _init_dist():
    yield


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("k,n", [(256, 272), (5120, 10240), (17408, 5120), (5120, 248320)])
@pytest.mark.parametrize("bias", [False, True])
@torch.inference_mode()
def test_same_rows_alone_packed_permuted_and_strided(k, n, bias):
    torch.manual_seed(481)
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * .01
    source = torch.randn(17, k * 2, device="cuda", dtype=torch.bfloat16)
    x = source[:, :k]  # Non-contiguous row stride is intentional.
    b = torch.randn(n, device="cuda", dtype=torch.bfloat16) if bias else None
    expected = torch.cat([invariant_linear(row, weight, b) for row in x.split(1)])
    for count in (2, 3, 8, 17):
        assert torch.equal(invariant_linear(x[:count], weight, b), expected[:count])
    order = torch.randperm(17, device="cuda")
    assert torch.equal(invariant_linear(x[order], weight, b), expected[order])
    reference = x.float() @ weight.float().T
    if b is not None:
        reference += b.float()
    torch.testing.assert_close(expected.float(), reference, rtol=.008, atol=.008)
