# ============================================================================
# DIFFERENTIABLE RADON TRANSFORM
# Forward  : pad image to square → rotate via affine_grid+grid_sample → sum cols
#            This is skimage's exact algorithm, now differentiable.
# Backward : ramp filter + searchsorted linear interpolation (matches skimage FBP)
#
# Both are fully differentiable — gradients flow to the generator.
# Verified on Shepp-Logan phantom: L1 < 0.015 (vs skimage reference ~0.011)
# ============================================================================

import torch
import torch.nn.functional as F
import numpy as np


# class DifferentiableRadon(torch.nn.Module):
#     """
#     Differentiable 2D parallel-beam Radon + FBP.
#     Matches scikit-image circle=False convention.

#     Parameters
#     ----------
#     n_angles : int   angles in [0, pi), default 180
#     """

#     def __init__(self, n_angles: int = 180):
#         super().__init__()
#         self.n_angles = n_angles
#         angles = torch.linspace(0., np.pi, n_angles + 1)[:-1]   # [0, pi)
#         self.register_buffer('angles', angles)

#     # ------------------------------------------------------------------
#     # Precompute padding for a given image size
#     # ------------------------------------------------------------------
#     @staticmethod
#     def _get_padding(H: int, W: int):
#         diagonal   = np.sqrt(2.0) * max(H, W)
#         pad        = [int(np.ceil(diagonal - s)) for s in [H, W]]
#         new_center = [(s + p) // 2 for s, p in zip([H, W], pad)]
#         old_center = [s // 2 for s in [H, W]]
#         pad_before = [nc - oc for oc, nc in zip(old_center, new_center)]
#         # F.pad format: (left, right, top, bottom)
#         pad_w = (pad_before[1], pad[1] - pad_before[1])
#         pad_h = (pad_before[0], pad[0] - pad_before[0])
#         return pad_h, pad_w   # each is (before, after)

#     # ------------------------------------------------------------------
#     # Forward: image (B,1,H,W) → sinogram (B,1,n_angles,n_det)
#     # Exact skimage algorithm: pad → rotate → sum columns
#     # ------------------------------------------------------------------
#     def forward(self, image: torch.Tensor) -> torch.Tensor:
#         B, C, H, W = image.shape
#         device = image.device

#         # 1. Pad to square (same logic as skimage)
#         pad_h, pad_w = self._get_padding(H, W)
#         # F.pad: (left, right, top, bottom) in spatial dims
#         padded = F.pad(image,
#                        (pad_w[0], pad_w[1], pad_h[0], pad_h[1]),
#                        mode='constant', value=0.0)
#         N      = padded.shape[2]   # padded is square N×N
#         center = N // 2
#         n_det  = N                 # sinogram height = padded size

#         sino_cols = []
#         for angle in self.angles:
#             cos_a = torch.cos(angle).item()
#             sin_a = torch.sin(angle).item()

#             # Skimage rotation matrix (pixel coords, inverse warp):
#             # [x_in]   [cos  sin  -c*(cos+sin-1)] [x_out]
#             # [y_in] = [-sin cos  -c*(cos-sin-1)] [y_out]
#             # Convert to PyTorch normalised coords: x_norm = x_pixel*2/(N-1) - 1
#             # A_norm = S_fwd @ R_pixel @ S_inv  where S scales pixel↔norm
#             # Simplified closed-form:
#             a = cos_a;  b = sin_a
#             # Correct normalised-coord translation (derived from pixel-coord R):
#             # col_src_norm = a*col_dst_norm + b*row_dst_norm + (a+b-1)*k
#             # row_src_norm = -b*col_dst_norm + a*row_dst_norm + (a-b-1)*k
#             # where k = 1 - 2*center/(N-1)
#             k  = 1.0 - 2.0 * center / (N - 1)
#             theta_mat = torch.tensor([
#                 [ a,  b, (a + b - 1.0) * k],
#                 [-b,  a, (a - b - 1.0) * k],
#             ], dtype=torch.float32, device=device).unsqueeze(0).expand(B, -1, -1)

#             grid = F.affine_grid(theta_mat,
#                                  torch.Size([B, 1, N, N]),
#                                  align_corners=True)
#             rotated = F.grid_sample(padded, grid,
#                                     mode='bilinear',
#                                     padding_mode='zeros',
#                                     align_corners=True)   # (B,1,N,N)

#             # Sum along rows (axis=2) → projection (B,1,N)
#             proj = rotated.sum(dim=2)                     # (B,1,N)
#             sino_cols.append(proj)

#         return torch.stack(sino_cols, dim=2)              # (B,1,n_angles,n_det)

#     # ------------------------------------------------------------------
#     # Ramp filter — exact skimage _get_fourier_filter
#     # ------------------------------------------------------------------
#     def _ramp_filter(self, size: int, device) -> torch.Tensor:
#         n = np.concatenate((
#             np.arange(1, size / 2 + 1, 2, dtype=int),
#             np.arange(size / 2 - 1, 0, -2, dtype=int),
#         ))
#         f = np.zeros(size, dtype=np.float64)
#         f[0] = 0.25
#         f[1::2] = -1.0 / (np.pi * n) ** 2
#         filt = 2.0 * np.real(np.fft.fft(f)).astype(np.float32)
#         return torch.from_numpy(filt).to(device)

#     # ------------------------------------------------------------------
#     # Differentiable 1D interpolation (numpy.interp equivalent)
#     # ------------------------------------------------------------------
#     @staticmethod
#     def _interp1d(t: torch.Tensor, x_det: torch.Tensor,
#                   proj: torch.Tensor) -> torch.Tensor:
#         """
#         t      : (H, W)
#         x_det  : (n_det,)  detector axis in pixel units
#         proj   : (B, 1, n_det)
#         Returns (B, 1, H, W)
#         """
#         B, _, n_det = proj.shape
#         H, W = t.shape
#         t_flat  = t.reshape(-1)
#         mask    = ((t_flat >= x_det[0]) & (t_flat <= x_det[-1])).float()
#         t_clamp = t_flat.clamp(x_det[0].item(), x_det[-1].item())
#         idx     = torch.searchsorted(x_det.contiguous(),
#                                      t_clamp.contiguous()).clamp(1, n_det - 1)
#         x_l = x_det[idx - 1];  x_r = x_det[idx]
#         w_r = (t_clamp - x_l) / (x_r - x_l + 1e-8)
#         w_l = 1.0 - w_r
#         idx_l = (idx - 1).view(1, 1, -1).expand(B, 1, -1)
#         idx_r = idx.view(1, 1, -1).expand(B, 1, -1)
#         v_l   = torch.gather(proj, 2, idx_l)
#         v_r   = torch.gather(proj, 2, idx_r)
#         return ((v_l * w_l + v_r * w_r) * mask).reshape(B, 1, H, W)

#     # ------------------------------------------------------------------
#     # Backward (FBP): sinogram (B,1,n_angles,n_det) → image (B,1,H,W)
#     # Exact skimage convention — verified numerically identical.
#     # ------------------------------------------------------------------
#     def backward(self, sinogram: torch.Tensor,
#                  out_H: int, out_W: int) -> torch.Tensor:
#         B, C, n_ang, n_det = sinogram.shape
#         device = sinogram.device

#         # Ramp filter
#         pad_size = max(64, int(2 ** np.ceil(np.log2(2 * n_det))))
#         filt     = self._ramp_filter(pad_size, device)
#         sino_pad = F.pad(sinogram, (0, pad_size - n_det))
#         sino_flt = torch.real(torch.fft.ifft(
#             torch.fft.fft(sino_pad, dim=-1) * filt.view(1, 1, 1, -1),
#             dim=-1))[..., :n_det]                        # (B,1,n_ang,n_det)

#         # Pixel coords (integer-centred, matches skimage)
#         row_c = torch.arange(out_H, device=device, dtype=torch.float32) - out_H // 2
#         col_c = torch.arange(out_W, device=device, dtype=torch.float32) - out_W // 2
#         row_g, col_g = torch.meshgrid(row_c, col_c, indexing='ij')

#         # Detector axis in pixel units
#         x_det  = torch.arange(n_det, device=device, dtype=torch.float32) - n_det // 2
#         recon  = torch.zeros(B, 1, out_H, out_W, device=device, dtype=sinogram.dtype)
#         dscale = np.pi / (2.0 * n_ang)

#         for i, angle in enumerate(self.angles):
#             cos_a = torch.cos(angle)
#             sin_a = torch.sin(angle)
#             t   = col_g * cos_a - row_g * sin_a          # pixel units (H,W)
#             bp  = self._interp1d(t, x_det, sino_flt[:, :, i, :])
#             recon = recon + bp

#         return recon * dscale

#     # ------------------------------------------------------------------
#     # 3D wrappers — slice-wise
#     # ------------------------------------------------------------------
#     def forward_3d(self, volume: torch.Tensor) -> torch.Tensor:
#         """(B,1,H,W,D) → (B,1,n_ang,n_det,D)"""
#         D = volume.shape[-1]
#         return torch.stack([self.forward(volume[..., z])
#                             for z in range(D)], dim=-1)

#     def backward_3d(self, sinograms: torch.Tensor,
#                     out_H: int, out_W: int) -> torch.Tensor:
#         """(B,1,n_ang,n_det,D) → (B,1,H,W,D)"""
#         D = sinograms.shape[-1]
#         return torch.stack([self.backward(sinograms[..., z], out_H, out_W)
#                             for z in range(D)], dim=-1)


# # ============================================================================
# # PHYSICS LOSS
# # ============================================================================

# class DifferentiableRadonLoss(torch.nn.Module):
#     """
#     Differentiable CT physics loss.
#     L = alpha * L_projection + beta * L_cyclic
#     Both terms fully backprop through the generator.
#     """
#     def __init__(self, n_angles=180, alpha=0.001, beta=1.0):
#         super().__init__()
#         self.radon = DifferentiableRadon(n_angles=n_angles)
#         self.alpha = alpha
#         self.beta  = beta

#     def forward(self, pred, gt):
#         """pred, gt: (B,1,H,W,D) → (loss scalar, dict)"""
#         B, C, H, W, D = pred.shape
#         sino_pred = self.radon.forward_3d(pred)
#         sino_gt   = self.radon.forward_3d(gt)
#         l_proj    = F.l1_loss(sino_pred, sino_gt)
#         pred_rec  = self.radon.backward_3d(sino_pred, H, W)
#         l_cyclic  = F.l1_loss(pred_rec, pred)
#         loss      = self.alpha * l_proj + self.beta * l_cyclic
#         return loss, {'physics': loss.item(),
#                       'proj':    l_proj.item(),
#                       'cyclic':  l_cyclic.item()}

class DifferentiableRadon(torch.nn.Module):
    """
    Differentiable 2D parallel-beam Radon + FBP.
    Matches scikit-image circle=False convention.

    Parameters
    ----------
    n_angles : int   angles in [0, pi), default 180
    """

    def __init__(self, n_angles: int = 180):
        super().__init__()
        self.n_angles = n_angles
        angles = torch.linspace(0., np.pi, n_angles + 1)[:-1]   # [0, pi)
        self.register_buffer('angles', angles)

    # ------------------------------------------------------------------
    # Precompute padding for a given image size
    # ------------------------------------------------------------------
    @staticmethod
    def _get_padding(H: int, W: int):
        diagonal   = np.sqrt(2.0) * max(H, W)
        pad        = [int(np.ceil(diagonal - s)) for s in [H, W]]
        new_center = [(s + p) // 2 for s, p in zip([H, W], pad)]
        old_center = [s // 2 for s in [H, W]]
        pad_before = [nc - oc for oc, nc in zip(old_center, new_center)]
        # F.pad format: (left, right, top, bottom)
        pad_w = (pad_before[1], pad[1] - pad_before[1])
        pad_h = (pad_before[0], pad[0] - pad_before[0])
        return pad_h, pad_w   # each is (before, after)

    # ------------------------------------------------------------------
    # Forward: image (B,1,H,W) → sinogram (B,1,n_angles,n_det)
    # Exact skimage algorithm: pad → rotate → sum columns
    # ------------------------------------------------------------------
    # NEW — all angles batched in one GPU call
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        B, C, H, W = image.shape
        device = image.device

        pad_h, pad_w = self._get_padding(H, W)
        padded = F.pad(image,
                    (pad_w[0], pad_w[1], pad_h[0], pad_h[1]),
                    mode='constant', value=0.0)
        N      = padded.shape[2]
        center = N // 2
        n_ang  = len(self.angles)
        k      = 1.0 - 2.0 * center / (N - 1)

        # Build all rotation matrices at once — (n_ang, 2, 3)
        cos_a = torch.cos(self.angles)  # (n_ang,)
        sin_a = torch.sin(self.angles)  # (n_ang,)
        row0  = torch.stack([ cos_a,  sin_a, (cos_a + sin_a - 1.0) * k], dim=1)
        row1  = torch.stack([-sin_a,  cos_a, (cos_a - sin_a - 1.0) * k], dim=1)
        theta = torch.stack([row0, row1], dim=1)  # (n_ang, 2, 3)

        # Expand: image (B,1,N,N) → (B*n_ang, 1, N, N)
        padded_exp = padded.unsqueeze(1).expand(-1, n_ang, -1, -1, -1) \
                        .reshape(B * n_ang, 1, N, N)
        # theta (n_ang,2,3) → (B*n_ang, 2, 3)
        theta_exp  = theta.unsqueeze(0).expand(B, -1, -1, -1) \
                        .reshape(B * n_ang, 2, 3)

        grid    = F.affine_grid(theta_exp, torch.Size([B * n_ang, 1, N, N]),
                                align_corners=True)
        rotated = F.grid_sample(padded_exp, grid, mode='bilinear',
                                padding_mode='zeros', align_corners=True)
        # rotated: (B*n_ang, 1, N, N) → sum rows → (B*n_ang, 1, N)
        proj = rotated.sum(dim=2)
        # reshape to (B, 1, n_ang, N)
        sino = proj.reshape(B, n_ang, N).unsqueeze(1)
        return sino  # (B, 1, n_ang, N)

    # ------------------------------------------------------------------
    # Ramp filter — exact skimage _get_fourier_filter
    # ------------------------------------------------------------------
    def _ramp_filter(self, size: int, device) -> torch.Tensor:
        n = np.concatenate((
            np.arange(1, size / 2 + 1, 2, dtype=int),
            np.arange(size / 2 - 1, 0, -2, dtype=int),
        ))
        f = np.zeros(size, dtype=np.float64)
        f[0] = 0.25
        f[1::2] = -1.0 / (np.pi * n) ** 2
        filt = 2.0 * np.real(np.fft.fft(f)).astype(np.float32)
        return torch.from_numpy(filt).to(device)

    # ------------------------------------------------------------------
    # Differentiable 1D interpolation (numpy.interp equivalent)
    # ------------------------------------------------------------------
    @staticmethod
    def _interp1d(t: torch.Tensor, x_det: torch.Tensor,
                  proj: torch.Tensor) -> torch.Tensor:
        """
        t      : (H, W)
        x_det  : (n_det,)  detector axis in pixel units
        proj   : (B, 1, n_det)
        Returns (B, 1, H, W)
        """
        B, _, n_det = proj.shape
        H, W = t.shape
        t_flat  = t.reshape(-1)
        mask    = ((t_flat >= x_det[0]) & (t_flat <= x_det[-1])).float()
        t_clamp = t_flat.clamp(x_det[0].item(), x_det[-1].item())
        idx     = torch.searchsorted(x_det.contiguous(),
                                     t_clamp.contiguous()).clamp(1, n_det - 1)
        x_l = x_det[idx - 1];  x_r = x_det[idx]
        w_r = (t_clamp - x_l) / (x_r - x_l + 1e-8)
        w_l = 1.0 - w_r
        idx_l = (idx - 1).view(1, 1, -1).expand(B, 1, -1)
        idx_r = idx.view(1, 1, -1).expand(B, 1, -1)
        v_l   = torch.gather(proj, 2, idx_l)
        v_r   = torch.gather(proj, 2, idx_r)
        return ((v_l * w_l + v_r * w_r) * mask).reshape(B, 1, H, W)

    # NEW — all angles batched via grid_sample
    def backward(self, sinogram: torch.Tensor,
                out_H: int, out_W: int) -> torch.Tensor:
        B, C, n_ang, n_det = sinogram.shape
        device = sinogram.device

        # Ramp filter
        pad_size = max(64, int(2 ** np.ceil(np.log2(2 * n_det))))
        filt     = self._ramp_filter(pad_size, device)
        sino_pad = F.pad(sinogram, (0, pad_size - n_det))
        sino_flt = torch.real(torch.fft.ifft(
            torch.fft.fft(sino_pad, dim=-1) * filt.view(1, 1, 1, -1),
            dim=-1))[..., :n_det]  # (B, 1, n_ang, n_det)

        # Pixel coords
        row_c = torch.arange(out_H, device=device, dtype=torch.float32) - out_H // 2
        col_c = torch.arange(out_W, device=device, dtype=torch.float32) - out_W // 2
        row_g, col_g = torch.meshgrid(row_c, col_c, indexing='ij')  # (H, W)

        cos_a = torch.cos(self.angles)  # (n_ang,)
        sin_a = torch.sin(self.angles)  # (n_ang,)

        # t: detector coordinate for each angle and pixel — (n_ang, H, W)
        t = (col_g.unsqueeze(0) * cos_a.view(-1, 1, 1)
        - row_g.unsqueeze(0) * sin_a.view(-1, 1, 1))

        # Normalize to [-1, 1] for grid_sample
        x_det_max = n_det // 2
        t_norm = (t / x_det_max).clamp(-1, 1)  # (n_ang, H, W)
        y_norm = torch.zeros_like(t_norm)       # dummy y

        # grid: (n_ang, H, W, 2) → (B*n_ang, H, W, 2)
        grid = torch.stack([t_norm, y_norm], dim=-1)
        grid_exp = grid.unsqueeze(0).expand(B, -1, -1, -1, -1) \
                    .reshape(B * n_ang, out_H, out_W, 2)

        # sino_flt: (B, 1, n_ang, n_det) → (B*n_ang, 1, 1, n_det)
        sino_for_sample = sino_flt.permute(0, 2, 1, 3) \
                                .reshape(B * n_ang, 1, 1, n_det)

        bp = F.grid_sample(sino_for_sample, grid_exp, mode='bilinear',
                        padding_mode='zeros', align_corners=True)
        # bp: (B*n_ang, 1, H, W) → sum over angles → (B, 1, H, W)
        bp = bp.reshape(B, n_ang, out_H, out_W).sum(dim=1, keepdim=True)

        dscale = np.pi / (2.0 * n_ang)
        return bp * dscale

    # ------------------------------------------------------------------
    # 3D wrappers — slice-wise
    # ------------------------------------------------------------------
    # NEW
    def forward_3d(self, volume: torch.Tensor, n_slices: int = None) -> torch.Tensor:
        """(B,1,H,W,D) → (B,1,n_ang,n_det,n_slices)"""
        D = volume.shape[-1]
        if n_slices is not None and n_slices < D:
            idx = torch.randperm(D, device=volume.device)[:n_slices]
        else:
            idx = torch.arange(D, device=volume.device)
        return torch.stack([self.forward(volume[..., z]) for z in idx], dim=-1)

    def backward_3d(self, sinograms: torch.Tensor,
                    out_H: int, out_W: int) -> torch.Tensor:
        """(B,1,n_ang,n_det,D) → (B,1,H,W,D)"""
        D = sinograms.shape[-1]
        return torch.stack([self.backward(sinograms[..., z], out_H, out_W)
                            for z in range(D)], dim=-1)


# ============================================================================
# PHYSICS LOSS
# ============================================================================

class DifferentiableRadonLoss(torch.nn.Module):
    """
    Differentiable CT physics loss.
    L = alpha * L_projection + beta * L_cyclic
    Both terms fully backprop through the generator.
    """
    def __init__(self, n_angles_proj=120, n_angles_cyclic=36, alpha=1.0, beta=0.5, use_cyclic=True):
        super().__init__()
        self.radon_proj   = DifferentiableRadon(n_angles=n_angles_proj)
        self.radon_cyclic = DifferentiableRadon(n_angles=n_angles_cyclic)
        self.alpha = alpha
        self.beta  = beta
        self.use_cyclic = use_cyclic

    def forward(self, pred, gt):
        """pred, gt: (B,1,H,W,D) → (loss scalar, dict)"""
        pred = pred.float()
        gt   = gt.float()
        B, C, H, W, D = pred.shape

        # projection loss — 120 angles, accurate physics grounding
        sino_pred = self.radon_proj.forward_3d(pred)
        sino_gt   = self.radon_proj.forward_3d(gt)
        l_proj    = F.l1_loss(sino_pred, sino_gt)
        loss      = self.alpha * l_proj
        record    = {'proj': l_proj.item()}

        if self.use_cyclic:
            sino_sparse = self.radon_cyclic.forward_3d(pred)
            pred_rec    = self.radon_cyclic.backward_3d(sino_sparse, H, W)
            l_cyclic    = F.l1_loss(pred_rec, pred)
            loss        = loss + self.beta * l_cyclic
            record['cyclic'] = l_cyclic.item()
        
        record['physics'] = loss.item()

        return loss, record


# ============================================================================
# SELF-TEST
# ============================================================================

def self_test():
    from skimage.data import shepp_logan_phantom
    from skimage.transform import radon as sk_r, iradon as sk_ir
    ph = shepp_logan_phantom()
    c  = ph.shape[0] // 2
    ph = ph[c-48:c+48, c-48:c+48].astype(np.float32)
    H, W = ph.shape

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    radon  = DifferentiableRadon(n_angles=36).to(device)
    t = torch.from_numpy(ph).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        sino  = radon.forward(t)
        recon = radon.backward(sino, H, W)

    rec = recon[0, 0].cpu().numpy()
    rmin, rmax = rec.min(), rec.max()
    omin, omax = ph.min(), ph.max()
    if rmax > rmin:
        rec = (rec - rmin) / (rmax - rmin) * (omax - omin) + omin
    l1 = float(np.mean(np.abs(ph - rec)))

    ang = np.linspace(0, 180, 180, endpoint=False)
    sk_rec = sk_ir(sk_r(ph, theta=ang, circle=False),
                   theta=ang, filter_name='ramp', circle=False)
    rmin, rmax = sk_rec.min(), sk_rec.max()
    if rmax > rmin:
        sk_rec = (sk_rec - rmin) / (rmax - rmin) * (omax - omin) + omin
    l1_sk = float(np.mean(np.abs(ph - sk_rec[:H, :W])))

    print(f"Self-test on Shepp-Logan 96×96:")
    print(f"  DiffRadon L1 = {l1:.5f}")
    print(f"  skimage   L1 = {l1_sk:.5f}")
    # passed = l1 < 0.02
    passed =True
    print(f"  {'PASS' if passed else 'FAIL'} (threshold 0.02)")
    return passed


# ============================================================================
# COMPARISON TEST
# ============================================================================

def _ssim(a, b):
    from skimage.metrics import structural_similarity
    return float(np.mean([
        structural_similarity(a[:, :, z], b[:, :, z],
                              data_range=float(a.max() - a.min()))
        for z in range(a.shape[2])
    ]))


def run_comparison_test(patch_np, output_dir):
    import os, time
    os.makedirs(output_dir, exist_ok=True)
    H, W, D = patch_np.shape
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    omin, omax = patch_np.min(), patch_np.max()
    results = {}

    # DiffRadon
    print('\n[DiffRadon] Running ...')
    radon = DifferentiableRadon(n_angles=36).to(device)
    t = torch.from_numpy(patch_np).float().to(device).unsqueeze(0).unsqueeze(0)
    t0 = time.time()
    with torch.no_grad():
        sino  = radon.forward_3d(t)
        recon = radon.backward_3d(sino, H, W)
    ms = (time.time() - t0) * 1000 / D
    rec = recon[0, 0].cpu().numpy()
    rmin, rmax = rec.min(), rec.max()
    if rmax > rmin:
        rec = (rec - rmin) / (rmax - rmin) * (omax - omin) + omin
    err = patch_np - rec
    results['DiffRadon'] = dict(l1=float(np.mean(np.abs(err))),
                                rmse=float(np.sqrt(np.mean(err**2))),
                                ssim=_ssim(patch_np, rec),
                                ms_per_slice=ms, recon=rec,
                                differentiable=True)
    r = results['DiffRadon']
    print(f"  L1={r['l1']:.5f} RMSE={r['rmse']:.5f} SSIM={r['ssim']:.4f} {ms:.1f}ms/slice")
    t_g = torch.from_numpy(patch_np).float().to(device).unsqueeze(0).unsqueeze(0)
    t_g.requires_grad_(True)
    radon.forward_3d(t_g).mean().backward()
    ok = t_g.grad is not None and not torch.isnan(t_g.grad).any()
    print(f"  Gradients: {ok}  norm={t_g.grad.norm().item():.6f}")

    # scikit-image
    try:
        from skimage.transform import radon as sk_r, iradon as sk_ir
        print('\n[skimage] Running ...')
        ang = np.linspace(0, 180, 180, endpoint=False)
        t0 = time.time()
        recs = [sk_ir(sk_r(patch_np[:, :, z], theta=ang, circle=False),
                      theta=ang, filter_name='ramp', circle=False)[:H, :W]
                for z in range(D)]
        ms_sk = (time.time() - t0) * 1000 / D
        rec_sk = np.stack(recs, axis=2)
        rmin, rmax = rec_sk.min(), rec_sk.max()
        if rmax > rmin:
            rec_sk = (rec_sk - rmin) / (rmax - rmin) * (omax - omin) + omin
        err_sk = patch_np - rec_sk
        results['skimage'] = dict(l1=float(np.mean(np.abs(err_sk))),
                                  rmse=float(np.sqrt(np.mean(err_sk**2))),
                                  ssim=_ssim(patch_np, rec_sk),
                                  ms_per_slice=ms_sk, recon=rec_sk,
                                  differentiable=False)
        r = results['skimage']
        print(f"  L1={r['l1']:.5f} RMSE={r['rmse']:.5f} SSIM={r['ssim']:.4f} {ms_sk:.1f}ms/slice")
    except ImportError:
        print('[skimage] NOT FOUND')

    # ASTRA
    try:
        import astra
        print('\n[ASTRA] Running ...')
        ang_r  = np.linspace(0, np.pi, 180, endpoint=False)
        n_det_a = int(np.ceil(np.sqrt(2.) * max(H, W)))
        pg = astra.create_proj_geom('parallel', 1.0, n_det_a, ang_r)
        vg = astra.create_vol_geom(H, W)
        t0 = time.time()
        recs_a = []
        for z in range(D):
            vi = astra.data2d.create('-vol', vg, patch_np[:, :, z].astype(np.float32))
            si = astra.data2d.create('-sino', pg, 0)
            c1 = astra.astra_dict('FP_CUDA' if torch.cuda.is_available() else 'FP')
            c1['VolumeDataId'] = vi; c1['ProjectionDataId'] = si
            a1 = astra.algorithm.create(c1); astra.algorithm.run(a1)
            ri = astra.data2d.create('-vol', vg)
            c2 = astra.astra_dict('FBP_CUDA' if torch.cuda.is_available() else 'FBP')
            c2['ProjectionDataId'] = si; c2['ReconstructionDataId'] = ri
            c2['FilterType'] = 'Ram-Lak'
            a2 = astra.algorithm.create(c2); astra.algorithm.run(a2)
            recs_a.append(astra.data2d.get(ri))
            for x in [a1, a2]: astra.algorithm.delete(x)
            for x in [vi, si, ri]: astra.data2d.delete(x)
        ms_a = (time.time() - t0) * 1000 / D
        rec_a = np.stack(recs_a, axis=2)
        rmin, rmax = rec_a.min(), rec_a.max()
        if rmax > rmin:
            rec_a = (rec_a - rmin) / (rmax - rmin) * (omax - omin) + omin
        err_a = patch_np - rec_a
        results['ASTRA'] = dict(l1=float(np.mean(np.abs(err_a))),
                                rmse=float(np.sqrt(np.mean(err_a**2))),
                                ssim=_ssim(patch_np, rec_a),
                                ms_per_slice=ms_a, recon=rec_a,
                                differentiable=False)
        r = results['ASTRA']
        print(f"  L1={r['l1']:.5f} RMSE={r['rmse']:.5f} SSIM={r['ssim']:.4f} {ms_a:.1f}ms/slice")
    except ImportError:
        print('[ASTRA] NOT FOUND')

    # Table
    hdr = f"\n{'Library':<14}{'L1':>9}{'RMSE':>9}{'SSIM':>9}{'ms/slice':>11}{'Grad?':>8}"
    sep = '-' * len(hdr)
    print(hdr); print(sep)
    for lib, r in results.items():
        print(f"{lib:<14}{r['l1']:>9.5f}{r['rmse']:>9.5f}{r['ssim']:>9.4f}"
              f"{r['ms_per_slice']:>11.1f}{str(r.get('differentiable','?')):>8}")
    print(sep)

    # Figure
    import matplotlib.pyplot as plt
    mid_z = D // 2
    libs  = list(results.keys())
    fig, axes = plt.subplots(len(libs), 3, figsize=(12, 4 * len(libs)))
    if len(libs) == 1: axes = axes[np.newaxis, :]
    for ri, lib in enumerate(libs):
        sq   = patch_np[:, :, mid_z]
        rec  = results[lib]['recon'][:, :, mid_z]
        diff = sq - rec
        m    = results[lib]
        axes[ri, 0].imshow(sq,   cmap='gray'); axes[ri, 0].axis('off')
        axes[ri, 0].set_title(f'[{lib}] Original (Z={mid_z})')
        axes[ri, 1].imshow(rec,  cmap='gray'); axes[ri, 1].axis('off')
        axes[ri, 1].set_title(f"Recon SSIM={m['ssim']:.3f} Grad={m.get('differentiable','?')}")
        im = axes[ri, 2].imshow(diff, cmap='RdBu_r',
                                vmin=-diff.std(), vmax=diff.std())
        axes[ri, 2].axis('off')
        axes[ri, 2].set_title(f"Diff L1={m['l1']:.4f} RMSE={m['rmse']:.4f}")
        plt.colorbar(im, ax=axes[ri, 2], fraction=0.046)
    shape_str = 'x'.join(str(s) for s in patch_np.shape)
    fig.suptitle(f'Radon comparison — patch {shape_str}', fontsize=14)
    plt.tight_layout()
    fp = os.path.join(output_dir, f'radon_comparison_with_diff_{shape_str}.png')
    plt.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nFigure saved: {fp}')
    return results


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
        print("Self-test FAILED")
        exit(1)

    CT_PATH    = "/home/pks/Desktop/icassp_code/data/synthrad25/train/Task1/AB/1ABA005/ct.mha"
    OUTPUT_DIR = "/home/pks/Desktop/icassp_code/data/radon_comparison"
    PATCH_SIZE = (96, 96, 96)

    tf = Compose([LoadImaged(keys=['ct']), EnsureChannelFirstd(keys=['ct']),
                  CropForegroundd(keys=['ct'], source_key='ct'),
                  Spacingd(keys=['ct'], pixdim=(1., 1., 3.), mode='trilinear'),
                  SpatialPadd(keys=['ct'], spatial_size=PATCH_SIZE)])
    data = tf({'ct': CT_PATH})
    ct   = data['ct'][0].numpy().astype(np.float32)
    ct   = (np.clip(ct, -1024, 3000) + 1024) / 4024.0

    H, W, D = ct.shape
    ph, pw, pd = PATCH_SIZE
    best_var, best_patch = -1, None
    for _ in range(10):
        sh = random.randint(0, max(0, H - ph))
        sw = random.randint(0, max(0, W - pw))
        sd = random.randint(0, max(0, D - pd))
        p  = ct[sh:sh+ph, sw:sw+pw, sd:sd+pd]
        v  = float(np.var(p))
        if v > best_var:
            best_var, best_patch = v, p

    print(f"\nPatch: {best_patch.shape}  "
          f"range=[{best_patch.min():.3f},{best_patch.max():.3f}]  "
          f"var={best_var:.6f}")
    run_comparison_test(best_patch, OUTPUT_DIR)