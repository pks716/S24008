import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor
import numpy as np


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

def convert_to_dict(x):
    if isinstance(x, torch.Tensor):
        return {'level_0': x}
    return x

class PixelLoss(nn.Module):
    def __init__(
        self,
        loss_type: str,
        weight: dict = None
    ):
        super().__init__()
        self.loss_type = loss_type
        self.weight = weight
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss()
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss()
    def forward(self, out, target, return_record = False):
        out = convert_to_dict(out)
        target = convert_to_dict(target)
        if self.weight is None:
            self.weight = {level: 1 for level in out.keys()}
        elif isinstance(self.weight, (int, float)):
            self.weight = {level: self.weight for level in out.keys()}
        loss = 0
        loss_record = {}
        for level in out.keys():
            o = out[level]
            if level in target:
                t = target[level]
            else:
                t = F.interpolate(target['level_0'], size = o.shape[2:], mode = 'nearest')
            l = self.loss_fn(o, t) * self.weight[level]
            loss += l
            loss_record[level] = l.item()
        if return_record:
            return loss, loss_record
        else:
            return loss

class BCEAdvLoss(nn.Module):
    def __init__(
        self,
        relativistic: bool = False,
        weight: dict = None
    ):
        super().__init__()
        self.relativistic = relativistic
        self.weight = weight
        self.loss_fn = nn.BCEWithLogitsLoss()
    def forward(self, fake, real = None, return_record = False):
        fake = convert_to_dict(fake)
        if self.relativistic:
            real = convert_to_dict(real)
        if self.weight is None:
            self.weight = {level: 1 for level in fake.keys()}
        elif isinstance(self.weight, (int, float)):
            self.weight = {level: self.weight for level in fake.keys()}
        loss = 0
        loss_record = {}
        for level in fake.keys():
            fake_ = fake[level]
            if self.relativistic:
                real_ = real[level].detach()
                l_real = self.loss_fn(real_ - torch.mean(fake_), torch.zeros_like(fake_))
                l_fake = self.loss_fn(fake_ - torch.mean(real_), torch.ones_like(fake_))
                l = (l_real + l_fake) / 2 * self.weight[level]
            else:
                l = self.loss_fn(fake_, torch.ones_like(fake_)) * self.weight[level]
            loss += l
            loss_record[level] = l.item()
        if return_record:
            return loss, loss_record
        else:
            return loss

class WGanAdvLoss(nn.Module):
    def __init__(
        self,
        softplus: bool = False,
        weight = None,
    ):
        super().__init__()
        self.weight = weight
        self.softplus = softplus
    def forward(self, fake, real = None, return_record = False):
        fake = convert_to_dict(fake)
        if self.weight is None:
            self.weight = {level: 1 for level in out.keys()}
        elif isinstance(self.weight, (int, float)):
            self.weight = {level: self.weight for level in out.keys()}
        loss = 0
        loss_record = {}
        for level in out.keys():
            fake_ = fake[level]
            if self.softplus:
                l = -fake_.mean() * self.weight[level]
            else:
                l = F.softplus(-fake_).mean() * self.weight[level]
            loss += l
            loss_record[level] = l.item()
        if return_record:
            return loss, loss_record
        else:
            return loss

class AdvLoss(nn.Module):
    def __init__(
        self,
        loss_fn
    ):
        super().__init__()
        self.loss_fn = loss_fn
    def forward(self, out):
        fake = out['disc_fake']
        real = out['disc_real']
        loss = 0
        loss_record = {}
        # disc loss
        l, l_record = self.loss_fn(fake, real, True)
        loss += l
        loss_record.update(l_record)
        return loss, loss_record

class PerceptionLoss3D(nn.Module):
    def __init__(
        self,
        feature_extractor: "nn.Module",
        loss_fn = nn.L1Loss(),
        channel_dim: int = 3,
        normalize_mean: list = [0.485, 0.456, 0.406],
        normalize_std: list = [0.229, 0.224, 0.225],
        separate_channel: bool = True,
        base_weight: float = 1.0,
        feature_weights: dict = None,
        views: str = ['axial', 'sagittal', 'coronal']
    ):
        super().__init__()
        self.views = views
        self.loss_fn = loss_fn
        self.feature_extractor = feature_extractor
        self.feature_extractor.eval()
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        self.channel_dim = channel_dim
        if normalize_mean is not None:
            c = len(normalize_mean)
            self.normalize_mean = torch.Tensor(normalize_mean).view(1,c,1,1,1)
            self.normalize_std = torch.Tensor(normalize_std).view(1,c,1,1,1)
        else:
            self.normalize_mean = normalize_mean
            self.normalize_std = normalize_std
        self.separate_channel = separate_channel
        self.base_weight = base_weight
    def forward(self, out, target, return_record = False):
        if isinstance(out, dict):
            out = out['level_0']
        if isinstance(target, dict):
            target = target['level_0']
        if self.normalize_mean is not None:
            device = out.device
            self.normalize_mean = self.normalize_mean.to(device)
            self.normalize_std = self.normalize_std.to(device)
            out = (out - self.normalize_mean) / (self.normalize_std + 1e-5)
            target = (target - self.normalize_mean) / (self.normalize_std + 1e-5)
        # init loss variable
        loss = 0
        # seperate channel dim into batch dim if needed
        b,c,h,w,z = out.shape
        if c != self.channel_dim or self.separate_channel:
            b_orig,c_orig,h,w,z = out.shape
            out = out.view(b_orig*c_orig,1,h,w,z)
            target = target.view(b_orig*c_orig,1,h,w,z)
            out = out.repeat(1,self.channel_dim,1,1,1)
            target = target.repeat(1,self.channel_dim,1,1,1)
        # Axial view
        if 'axial' in self.views:
            # make 3D->2D (axial view)
            b,c,h,w,z = out.shape
            o = out.permute(0,4,1,2,3).reshape(b*z,c,h,w)
            t = target.permute(0,4,1,2,3).reshape(b*z,c,h,w)
            # make channel dimension feasible to perception model
            o_features = self.feature_extractor(o)
            t_features = self.feature_extractor(t)
            for key in o_features.keys():
                b_z,c,h,w = o_features[key].shape
                z = b_z // b
                o_feat = o_features[key].reshape(b,z,c,h,w).permute(0,2,3,4,1) # make b,c,h,w,z again
                t_feat = t_features[key].reshape(b,z,c,h,w).permute(0,2,3,4,1) # make b,c,h,w,z again
                loss += self.loss_fn(o_feat, t_feat) / self.base_weight
        # sagittal view
        if 'sagittal' in self.views:
            # make 3D->2D (sagittal view)
            b,c,h,w,z = out.shape
            o = out.permute(0,3,1,2,4).reshape(b*w,c,h,z)
            t = target.permute(0,3,1,2,4).reshape(b*w,c,h,z)
            # make channel dimension feasible to perception model
            o_features = self.feature_extractor(o)
            t_features = self.feature_extractor(t)
            for key in o_features.keys():
                b_w,c,h,z = o_features[key].shape
                w = b_w // b
                o_feat = o_features[key].reshape(b,w,c,h,z).permute(0,2,3,1,4) # make b,c,h,w,z again
                t_feat = t_features[key].reshape(b,w,c,h,z).permute(0,2,3,1,4) # make b,c,h,w,z again
                loss += self.loss_fn(o_feat, t_feat) / self.base_weight
        if 'coronal' in self.views:
            # Coronal view
            # make 3D->2D (coronal view)
            b,c,h,w,z = out.shape
            o = out.permute(0,2,1,3,4).reshape(b*h,c,w,z)
            t = target.permute(0,2,1,3,4).reshape(b*h,c,w,z)
            # make channel dimension feasible to perception model
            o_features = self.feature_extractor(o)
            t_features = self.feature_extractor(t)
            for key in o_features.keys():
                b_h,c,w,z = o_features[key].shape
                h = b_h // b
                o_feat = o_features[key].reshape(b,h,c,w,z).permute(0,2,1,3,4) # make b,c,h,w,z again
                t_feat = t_features[key].reshape(b,h,c,w,z).permute(0,2,1,3,4) # make b,c,h,w,z again
                loss += self.loss_fn(o_feat, t_feat) / self.base_weight
            loss = loss / len(self.views)
            loss_record = loss.item()
            if return_record:
                return loss, loss_record
            else:
                return loss

def define_pretrained(model_name):
    if model_name == 'vgg19':
        pretrained = torchvision.models.vgg19(weights = torchvision.models.vgg.VGG19_Weights.IMAGENET1K_V1)
    elif model_name == 'resnet152':
        pretrained = torchvision.models.resnet152(weights = torchvision.models.resnet.ResNet152_Weights)
    elif model_name == 'inception_v3':
        pretrained = torchvision.models.inception_v3(weights = torchvision.models.inception.Inception_V3_Weights)
    elif model_name == 'resnet50':
        pretrained = torchvision.models.resnet50(weights=None)  # no imagenet weights
        checkpoint = torch.load('/home/pks/Desktop/icassp_code/codes/RadImageNet_pytorch/ResNet50.pt',
                                map_location='cpu')
        key_map = {
        'backbone.0': 'conv1',
        'backbone.1': 'bn1',
        'backbone.4': 'layer1',
        'backbone.5': 'layer2',
        'backbone.6': 'layer3',
        'backbone.7': 'layer4',
        }
        new_state_dict = {}
        for k, v in checkpoint.items():
            new_key = k
            for old_prefix, new_prefix in key_map.items():
                if k.startswith(old_prefix):
                    new_key = k.replace(old_prefix, new_prefix, 1)
                    break
            new_state_dict[new_key] = v
        pretrained.load_state_dict(new_state_dict, strict=False)
    
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")
    return pretrained

class GeneratorLoss(nn.Module):
    def __init__(
        self,
        pixel_loss_fn = None,
        perception_loss_fn = None,
        adv_loss_fn = None,
        physics_loss_fn = None,
        weight: dict = None
    ):
        super().__init__()
        self.pixel_loss_fn = pixel_loss_fn
        self.perception_loss_fn = perception_loss_fn
        self.adv_loss_fn = adv_loss_fn
        self.physics_loss_fn = physics_loss_fn
        self.weight = weight if weight is not None else {}

    def forward(self, out, target):
        loss = 0
        loss_record = {}
        # pixel loss
        if self.pixel_loss_fn is not None:
            o = out['out']
            w = self.weight['pixel'] if 'pixel' in self.weight else 1
            l, l_record = self.pixel_loss_fn(o, target, True)
            loss += l * w
            loss_record['pixel'] = l_record
        # perception loss
        if self.perception_loss_fn is not None:
            o = out['out']
            w = self.weight['perception'] if 'perception' in self.weight else 1
            l, l_record = self.perception_loss_fn(o, target, True)
            loss += l * w
            loss_record['perception'] = l_record
        # adv loss
        if self.adv_loss_fn is not None:
            w = self.weight['adv'] if 'adv' in self.weight else 1
            l, l_record = self.adv_loss_fn(out)
            loss += l * w
            loss_record['adv'] = l_record
        
        # physics loss
        if self.physics_loss_fn is not None:
            o = out['out']['level_0'] if isinstance(out['out'], dict) else out['out']
            t = target['level_0'] if isinstance(target, dict) else target
            w = self.weight['physics'] if 'physics' in self.weight else 1
            l, l_record = self.physics_loss_fn(o, t)
            loss += l * w
            loss_record['physics'] = l_record
        return loss, loss_record

def build_generator_loss(opts):
    pixel_loss = None
    perception_loss = None
    adv_loss = None
    tv_loss = None
    weight_by_loss = {}
    # get settings
    pixel_loss_opt = opts.get('pixel_loss')
    perception_loss_opt = opts.get('perception_loss')
    adv_loss_opt = opts.get('adv_loss')
    # pixel loss
    if pixel_loss_opt is not None:
        weight_by_level = pixel_loss_opt.get('weight_by_level')
        params = pixel_loss_opt['params']
        pixel_loss = PixelLoss(weight = weight_by_level, **params)
        weight_by_loss['pixel'] = pixel_loss_opt['weight']
    # perception loss
    if perception_loss_opt is not None: # only applied to the final output level
        loss_type = perception_loss_opt['type']
        params = perception_loss_opt['params']
        model_name = params['model_name']
        return_nodes = params['return_nodes']
        channel_dim = 3
        separate_channel = True
        base_weight = len(return_nodes)
        pretrained = define_pretrained(model_name).eval()
        feature_extractor = create_feature_extractor(pretrained, return_nodes)
        if loss_type == '3d':
            perception_loss = PerceptionLoss3D(feature_extractor, nn.L1Loss(), channel_dim = channel_dim, separate_channel = separate_channel, base_weight = base_weight)
        weight_by_loss['perception'] = perception_loss_opt['weight']
    # adv loss
    if adv_loss_opt is not None:
        weight_by_level = adv_loss_opt.get('weight_by_level')
        params = adv_loss_opt['params']
        loss_type = params['loss_type']
        if loss_type == 'bce':
            relativistic = params['relativistic']
            loss_fn = BCEAdvLoss(relativistic = relativistic, weight = weight_by_level)
        elif loss_type == 'wgan':
            softplus = params.get('softplus')
            if softplus is None:
                softplus = False
            loss_fn = WGanAdvLoss(softplus = softplus, weight = weight_by_level)
        adv_loss = AdvLoss(loss_fn)
        weight_by_loss['adv'] = adv_loss_opt['weight']

    # physics loss
    physics_loss = None
    physics_loss_opt = opts.get('physics_loss')
    if physics_loss_opt is not None:
        physics_loss = DifferentiableRadonLoss(
            n_angles_proj   = physics_loss_opt.get('n_angles_proj', 90),
            n_angles_cyclic = physics_loss_opt.get('n_angles_cyclic', 36),
            alpha           = physics_loss_opt.get('alpha', 1.0),
            beta            = physics_loss_opt.get('beta', 1.0),
            use_cyclic      = physics_loss_opt.get('use_cyclic', True),
        )
        weight_by_loss['physics'] = physics_loss_opt.get('weight', 0.1)
    return GeneratorLoss(pixel_loss_fn = pixel_loss, perception_loss_fn = perception_loss, adv_loss_fn = adv_loss, physics_loss_fn = physics_loss, weight = weight_by_loss)