# ============================================================================
# MRI K-SPACE PHYSICS LOSS — Pure PyTorch, Fully Differentiable
# ============================================================================
#
# Design philosophy — exact analog of the CT Radon cyclic loss:
#
#   CT:  pred → Radon(pred) → FBP → compare to pred
#        Non-trivial because FBP(Radon(x)) ≠ x on finite patches
#        Patch boundary + ramp filter approximation makes this lossy
#
#   MRI: pred → FFT(pred) → undersample(M) → iFFT → compare to pred
#        Non-trivial because iFFT(M*FFT(x)) ≠ x when M drops k-space lines
#        Missing lines cause aliasing — same truncation physics as CT
#
# Key fix vs previous version:
#   FIXED mask → generator learns to avoid specific frequencies (wrong)
#   RANDOM mask per forward pass → generator learns general k-space consistency
#
# This is the correct analog because:
#   CT patches change location each iteration → different projections truncated
#   MRI masks change each iteration → different k-space lines missing
#   Both enforce: "image must be consistent with ANY plausible acquisition"
#
# Loss terms:
#   L_cyclic = ||iFFT(M * FFT(pred)) - pred||_1      (acquisition consistency)
#   L_kspace = ||M*FFT(pred) - M*FFT(gt)||_1         (measured data fidelity)
#   L_total  = alpha * L_kspace + beta * L_cyclic
#
# Note on L_kspace:
#   Only compare SAMPLED k-space lines (multiply both sides by M)
#   Unsampled lines are physically unknown — penalising them is not motivated
# ============================================================================

import torch
import torch.nn.functional as F
import numpy as np


def _sample_random_mask(H: int, W: int,
                         acceleration: float = 4.0,
                         centre_fraction: float = 0.08,
                         device=None) -> torch.Tensor:
    """
    Sample a random Cartesian undersampling mask for one forward pass.
    Called fresh every forward pass — different mask each time.

    Cartesian sampling: 1D mask along phase-encode (W) direction,
    broadcast across frequency-encode (H) direction.
    Centre of k-space always fully sampled (low freq = tissue contrast).

    Returns
    -------
    mask : (1, H, W) float tensor, values in {0, 1}
    """
    mask = torch.zeros(W, device=device)

    # Always keep centre lines (low frequencies — critical for image contrast)
    n_centre = max(1, int(W * centre_fraction))
    centre   = W // 2
    mask[centre - n_centre // 2: centre + n_centre // 2] = 1.0

    # Randomly sample remaining lines to reach target acceleration
    n_total  = max(1, W // int(acceleration))
    n_random = max(0, n_total - n_centre)
    outer    = (mask == 0).nonzero(as_tuple=True)[0]
    if n_random > 0 and len(outer) > 0:
        perm     = torch.randperm(len(outer), device=device)
        selected = outer[perm[:min(n_random, len(outer))]]
        mask[selected] = 1.0

    # (W,) → (1, H, W): same column pattern applied to all rows
    mask_2d = mask.unsqueeze(0).expand(H, -1)   # (H, W)
    return mask_2d.unsqueeze(0)                  # (1, H, W)


class KSpaceLoss(torch.nn.Module):
    """
    MRI k-space physics loss — correct analog of CT Radon cyclic loss.

    Uses a fresh random Cartesian undersampling mask every forward pass
    so the generator cannot overfit to any specific sampling pattern.

    Parameters
    ----------
    acceleration    : undersampling R-factor (default 4 = ~25% sampled)
    centre_fraction : fraction of centre k-space always sampled
    alpha           : weight for sampled k-space fidelity loss
    beta            : weight for cyclic consistency loss
    norm            : FFT normalisation ('ortho' keeps scale stable)
    """

    def __init__(self, acceleration: float = 4.0,
                 centre_fraction: float = 0.08,
                 alpha: float = 0.1, beta: float = 1.0,
                 norm: str = 'ortho'):
        super().__init__()
        self.acceleration    = acceleration
        self.centre_fraction = centre_fraction
        self.alpha           = alpha
        self.beta            = beta
        self.norm            = norm

    # ------------------------------------------------------------------
    # FFT / iFFT helpers
    # ------------------------------------------------------------------
    def _fft2(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,1,H,W) → complex (B,1,H,W), centred k-space"""
        return torch.fft.fftshift(
            torch.fft.fft2(x, norm=self.norm, dim=(-2, -1)),
            dim=(-2, -1)
        )

    def _ifft2(self, k: torch.Tensor) -> torch.Tensor:
        """k: complex (B,1,H,W) → real magnitude (B,1,H,W)"""
        return torch.abs(
            torch.fft.ifft2(
                torch.fft.ifftshift(k, dim=(-2, -1)),
                norm=self.norm, dim=(-2, -1)
            )
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, pred: torch.Tensor,
                gt: torch.Tensor) -> tuple:
        """
        pred, gt : (B, 1, H, W, D)
        Returns  : (loss scalar, log dict)
        """
        B, C, H, W, D = pred.shape
        device = pred.device

        # Sample ONE mask for this forward pass — same mask across all
        # slices and batch items (consistent acquisition simulation)
        M = _sample_random_mask(H, W, self.acceleration,
                                 self.centre_fraction, device)   # (1,H,W)

        l_kspace = pred.new_zeros(1)
        l_cyclic = pred.new_zeros(1)

        for z in range(D):
            pred_sl = pred[..., z]   # (B,1,H,W)
            gt_sl   = gt[..., z]

            k_pred = self._fft2(pred_sl)   # complex (B,1,H,W)
            k_gt   = self._fft2(gt_sl)

            # ── Sampled k-space fidelity ─────────────────────────────
            # Only compare lines that would actually be measured
            # Unsampled lines are physically unknown — don't penalise them
            l_kspace = l_kspace + F.l1_loss(
                (k_pred * M).abs(),
                (k_gt   * M).abs()
            )

            # ── Cyclic consistency ───────────────────────────────────
            # pred → k-space → drop unsampled lines → reconstruct → compare
            # Non-trivial (≠0) because missing lines cause aliasing
            # Exact analog of CT: pred → Radon → FBP → compare
            k_under  = k_pred * M
            pred_rec = self._ifft2(k_under)   # (B,1,H,W) real
            l_cyclic = l_cyclic + F.l1_loss(pred_rec, pred_sl)

        l_kspace = l_kspace / D
        l_cyclic = l_cyclic / D
        loss     = self.alpha * l_kspace + self.beta * l_cyclic

        return loss, {
            'kspace_total':    loss.item(),
            'kspace_fidelity': l_kspace.item(),
            'cyclic':          l_cyclic.item(),
            'mask_density':    M.float().mean().item(),
        }

    # ------------------------------------------------------------------
    # Visualise — uses fixed seed for reproducible debugging output
    # ------------------------------------------------------------------
    def visualise(self, pred: torch.Tensor, output_path: str = None):
        """pred : (B, 1, H, W, D)"""
        import matplotlib.pyplot as plt

        z      = pred.shape[-1] // 2
        device = pred.device
        sl     = pred[0, 0, :, :, z].detach()
        H, W   = sl.shape

        torch.manual_seed(0)
        M = _sample_random_mask(H, W, self.acceleration,
                                 self.centre_fraction, device)

        with torch.no_grad():
            k    = self._fft2(sl.unsqueeze(0).unsqueeze(0))[0, 0]
            k_u  = k * M[0]
            rec  = self._ifft2(k_u.unsqueeze(0).unsqueeze(0))[0, 0]

        sl = sl.cpu(); k = k.cpu(); k_u = k_u.cpu(); rec = rec.cpu()

        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        axes[0].imshow(sl.numpy(),                           cmap='gray')
        axes[0].set_title('Original slice')
        axes[1].imshow(torch.log(k.abs()   + 1e-8).numpy(), cmap='hot')
        axes[1].set_title('K-space (log mag)')
        axes[2].imshow(M[0].cpu().float().numpy(),           cmap='gray')
        axes[2].set_title(f'Random mask (R={self.acceleration}x)')
        axes[3].imshow(torch.log(k_u.abs() + 1e-8).numpy(), cmap='hot')
        axes[3].set_title('Undersampled k-space')
        axes[4].imshow(rec.numpy(),                          cmap='gray')
        axes[4].set_title('Cyclic reconstruction')
        for ax in axes: ax.axis('off')
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved: {output_path}")
        else:
            plt.show()


# ============================================================================
# SELF-TEST
# ============================================================================

def self_test():
    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, H, W, D  = 1, 96, 96, 8
    loss_fn     = KSpaceLoss(acceleration=4.0).to(device)

    pred = torch.rand(B, 1, H, W, D, device=device, requires_grad=True)
    gt   = torch.rand(B, 1, H, W, D, device=device)
    loss, logs = loss_fn(pred, gt)
    loss.backward()

    grad_ok = pred.grad is not None and not torch.isnan(pred.grad).any()

    # FFT round-trip
    x     = torch.rand(1, 1, H, W, device=device)
    x_rec = loss_fn._ifft2(loss_fn._fft2(x))
    rt    = (x - x_rec).abs().max().item()

    # Mask varies across calls (random, not fixed)
    torch.manual_seed(1); m1 = _sample_random_mask(H, W, 4.0, 0.08, device)
    torch.manual_seed(2); m2 = _sample_random_mask(H, W, 4.0, 0.08, device)
    mask_varies = not torch.equal(m1, m2)

    # k-space fidelity should be ~0 when pred == gt
    pred2 = torch.rand(B, 1, H, W, D, device=device)
    _, logs2 = loss_fn(pred2, pred2)

    print(f"\nSelf-test:")
    print(f"  Loss:              {loss.item():.6f}  (finite, >0)")
    print(f"  K-space fidelity:  {logs['kspace_fidelity']:.6f}")
    print(f"  Cyclic:            {logs['cyclic']:.6f}")
    print(f"  Mask density:      {logs['mask_density']*100:.1f}%  (~{100/4:.0f}% expected)")
    print(f"  Gradients OK:      {grad_ok}  norm={pred.grad.norm().item():.6f}")
    print(f"  FFT roundtrip:     {rt:.2e}  (should be ~1e-6)")
    print(f"  Mask varies:       {mask_varies}  (should be True)")
    print(f"  loss(x,x) k-space: {logs2['kspace_fidelity']:.2e}  (should be ~0)")

    passed = (grad_ok and loss.item() > 0 and rt < 1e-4
              and mask_varies and logs2['kspace_fidelity'] < 1e-5)
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


# ============================================================================
# COMPARISON TEST
# ============================================================================

def run_comparison_test(mr_patch_np: np.ndarray, output_dir: str):
    import os, time
    os.makedirs(output_dir, exist_ok=True)
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    H, W, D  = mr_patch_np.shape
    loss_fn  = KSpaceLoss(acceleration=4.0, alpha=0.1, beta=1.0).to(device)

    t = torch.from_numpy(mr_patch_np).float().to(device).unsqueeze(0).unsqueeze(0)
    t.requires_grad_(True)

    t0 = time.time()
    loss, logs = loss_fn(t, t)
    elapsed = (time.time() - t0) * 1000
    loss.backward()
    grad_ok = t.grad is not None and not torch.isnan(t.grad).any()

    print(f"\n[KSpaceLoss] On real MRI patch {mr_patch_np.shape}:")
    print(f"  K-space fidelity (pred==gt): {logs['kspace_fidelity']:.2e}  (should be ~0)")
    print(f"  Cyclic loss:                 {logs['cyclic']:.6f}  (non-zero, aliasing)")
    print(f"  Mask density:                {logs['mask_density']*100:.1f}%")
    print(f"  Time: {elapsed:.1f}ms  |  Gradients: {grad_ok}  norm={t.grad.norm().item():.6f}")

    viz = os.path.join(output_dir, f'kspace_vis_{H}x{W}x{D}.png')
    loss_fn.visualise(t.detach(), output_path=viz)
    return logs


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    import random
    from monai.transforms import (LoadImaged, EnsureChannelFirstd,
                                   CropForegroundd, Spacingd, SpatialPadd, Compose)

    print("=" * 50)
    print("SELF-TEST")
    print("=" * 50)
    if not self_test():
        print("Self-test FAILED"); exit(1)

    MR_PATH    = "/home/pks/Desktop/icassp_code/data/SynthRad2025/Task1/AB/1ABA005/mr.mha"
    OUTPUT_DIR = "/home/pks/Desktop/icassp_code/data/"
    PATCH_SIZE = (96, 96, 96)

    tf = Compose([LoadImaged(keys=['mr']), EnsureChannelFirstd(keys=['mr']),
                  CropForegroundd(keys=['mr'], source_key='mr'),
                  Spacingd(keys=['mr'], pixdim=(1., 1., 2.5), mode='trilinear'),
                  SpatialPadd(keys=['mr'], spatial_size=PATCH_SIZE)])
    data = tf({'mr': MR_PATH})
    mr   = data['mr'][0].numpy().astype(np.float32)
    mr   = (mr - mr.min()) / (mr.max() - mr.min() + 1e-8)

    H, W, D = mr.shape; ph, pw, pd = PATCH_SIZE
    best_var, best_patch = -1, None
    for _ in range(10):
        sh = random.randint(0, max(0, H-ph))
        sw = random.randint(0, max(0, W-pw))
        sd = random.randint(0, max(0, D-pd))
        p  = mr[sh:sh+ph, sw:sw+pw, sd:sd+pd]
        v  = float(np.var(p))
        if v > best_var: best_var, best_patch = v, p

    print(f"\nPatch: {best_patch.shape}  range=[{best_patch.min():.3f},{best_patch.max():.3f}]")
    run_comparison_test(best_patch, OUTPUT_DIR)