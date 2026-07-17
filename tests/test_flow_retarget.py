"""Latent rectified-flow head: shapes, gradients, training convergence, sampler, guidance."""

import torch

from snmr.flow import (
    GuidanceConfig,
    LatentFlowConfig,
    LatentFlowNet,
    PhysicsCostConfig,
    flow_time_embedding,
    make_decoded_physics_cost,
    rectified_flow_loss,
    sample_flow,
    soft_stance_weights,
)


def _small_cfg(latent_dim=16):
    return LatentFlowConfig(latent_dim=latent_dim, hidden_dim=32, num_layers=2, num_heads=2,
                            time_embed_dim=16)


def test_time_embedding_shape_and_range():
    u = torch.tensor([0.0, 0.5, 1.0])
    emb = flow_time_embedding(u, 16)
    assert emb.shape == (3, 16)
    assert torch.isfinite(emb).all()
    assert emb.abs().max() <= 1.0 + 1e-6


def test_forward_shape_and_zero_init():
    torch.manual_seed(0)
    net = LatentFlowNet(_small_cfg())
    z = torch.randn(2, 8, 16)
    cond = torch.randn(2, 8, 16)
    u = torch.rand(2)
    v = net(z, u, cond)
    assert v.shape == (2, 8, 16)
    # zero-initialized output head: the untrained field is exactly zero everywhere
    assert torch.allclose(v, torch.zeros_like(v))


def test_untrained_sampler_returns_noise_unchanged():
    torch.manual_seed(0)
    net = LatentFlowNet(_small_cfg())
    cond = torch.randn(1, 6, 16)
    z0 = torch.randn(1, 6, 16)
    z1 = sample_flow(net, cond, num_steps=4, z0=z0)
    assert torch.allclose(z1, z0)


def test_loss_backward_reaches_all_parameters():
    torch.manual_seed(0)
    net = LatentFlowNet(_small_cfg())
    z1 = torch.randn(3, 8, 16)
    cond = torch.randn(3, 8, 16)
    loss = rectified_flow_loss(net, z1, cond)
    assert loss.item() > 0
    loss.backward()
    for name, p in net.named_parameters():
        assert p.grad is not None, name


def test_cfm_learns_constant_offset_transport():
    """cond → cond + delta is the simplest conditional transport; CFM must drive samples there."""
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    net = LatentFlowNet(_small_cfg(latent_dim=8))
    delta = torch.linspace(-1.0, 1.0, 8)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    for _ in range(400):
        cond = torch.randn(8, 4, 8, generator=gen)
        z1 = cond + delta
        opt.zero_grad()
        loss = rectified_flow_loss(net, z1, cond, generator=gen)
        loss.backward()
        opt.step()
    cond = torch.randn(4, 4, 8, generator=gen)
    z0 = torch.randn(4, 4, 8, generator=gen)
    z_hat = sample_flow(net, cond, num_steps=50, z0=z0)
    err = (z_hat - (cond + delta)).square().mean().item()
    base = (z0 - (cond + delta)).square().mean().item()
    assert err < 0.2 * base, f"sampled endpoint error {err} vs noise baseline {base}"


def test_guidance_moves_sample_toward_lower_cost():
    """With a quadratic cost pulling toward a point, guided samples must land closer to it."""
    torch.manual_seed(0)
    net = LatentFlowNet(_small_cfg(latent_dim=8))  # zero field: dynamics are guidance-only
    target = torch.full((1, 4, 8), 2.0)

    def cost(z_hat):
        return (z_hat - target).square().mean()

    cond = torch.zeros(1, 4, 8)
    z0 = torch.zeros(1, 4, 8)
    unguided = sample_flow(net, cond, num_steps=20, z0=z0)
    guided = sample_flow(
        net, cond, num_steps=20, z0=z0,
        guidance=GuidanceConfig(cost_fn=cost, alpha_start=5.0, alpha_end=5.0),
    )
    assert cost(guided).item() < cost(unguided).item()
    # per-element clamp bounds a single step's guidance displacement by h * alpha * grad_clamp
    assert (guided - z0).abs().max() <= 5.0 * 0.2 + 1e-6


def test_soft_stance_weights_monotone_and_bounded():
    h = torch.tensor([[[-0.01, 0.0, 0.03, 0.10]]])
    w = soft_stance_weights(h, height_threshold=0.03, temperature=0.01)
    assert w.shape == h.shape
    assert (w >= 0).all() and (w <= 1).all()
    flat = w.flatten()
    assert (flat[:-1] >= flat[1:]).all()  # deeper/lower feet get more stance weight
    assert flat[0] > 0.9 and flat[-1] < 0.1


def test_physics_cost_gradient_flows_to_latent():
    torch.manual_seed(0)
    lin = torch.nn.Linear(8, 2 * 3 + 4)  # stand-in frozen decoder: latent → 2 feet + 4 dof

    def decode_fn(z):
        out = lin(z)
        B, T, _ = out.shape
        return out[..., :6].reshape(B, T, 2, 3), out[..., 6:]

    cost = make_decoded_physics_cost(
        decode_fn, PhysicsCostConfig(fps=30.0)
    )
    z = torch.randn(1, 6, 8, requires_grad=True)
    value = cost(z)
    assert value.item() >= 0
    (grad,) = torch.autograd.grad(value, z)
    assert grad.shape == z.shape
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0


def test_physics_cost_stance_modes():
    """Oracle weights and gated modes reweight the skate term without breaking gradients."""
    torch.manual_seed(0)
    lin = torch.nn.Linear(8, 2 * 3 + 4)

    def decode_fn(z):
        out = lin(z)
        B, T, _ = out.shape
        return out[..., :6].reshape(B, T, 2, 3), out[..., 6:]

    cfg = PhysicsCostConfig(fps=30.0)
    z = torch.randn(1, 6, 8, requires_grad=True)
    oracle = torch.zeros(1, 6, 2)
    oracle[:, :3, 0] = 1.0
    gate = torch.ones(1, 6, 2)
    for cost in (
        make_decoded_physics_cost(decode_fn, cfg, stance_weights=oracle),
        make_decoded_physics_cost(decode_fn, cfg, stance_gate=gate),
    ):
        value = cost(z)
        (grad,) = torch.autograd.grad(value, z)
        assert torch.isfinite(grad).all()


def test_sampler_reproducible_with_explicit_noise():
    torch.manual_seed(0)
    net = LatentFlowNet(_small_cfg())
    for p in net.parameters():
        p.data.normal_(0, 0.02)
    cond = torch.randn(1, 5, 16)
    z0 = torch.randn(1, 5, 16)
    a = sample_flow(net, cond, num_steps=8, z0=z0)
    b = sample_flow(net, cond, num_steps=8, z0=z0)
    assert torch.allclose(a, b)
    assert not torch.allclose(a, z0)  # non-zero field actually transports
