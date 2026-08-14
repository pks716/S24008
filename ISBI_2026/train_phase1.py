
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from typing import Dict, Tuple
import numpy as np
from torchvision.models import convnext_tiny

from train_dataloader import create_contrastive_dataloaders

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data paths
train_data_root = ""
val_data_root = ""
logdir = ""
os.makedirs(logdir, exist_ok=True)

# Training hyperparameters
num_epochs = 1000  #better to use more number of epochs with higher capcity model
lr = 3e-4
weight_decay = 1e-4
num_patients_per_batch = 32 
num_slices = 1
target_size = (256, 256)

# Contrastive learning parameters
temperature = 0.07
emb_dim = 256  #A key parameter that needs to be tuned according to model (128 works, 256 and 512 may give better results)

# Loss weights
lambda_anatomy = 1.0   # Weight for anatomy contrastive loss
lambda_contrast = 1.0  # Weight for contrast contrastive loss
lambda_bank_alignment = 0.1  # Weight for bank alignment loss
lambda_modality_adv = 1.0   # Weight for adversarial modality loss
lambda_decor = 0.1         # Weight for decorrelation loss
lambda_proto_orth = 0.5  # start small (0.05-0.2)



modalities = ["t1", "t1ce", "t2", "flair"]

# ==============================================================================
# MODEL COMPONENTS (Adapted from CACO)
# ==============================================================================

class ConvNeXtEncoder(nn.Module):
    """
    ConvNeXt encoder.
    Extracts multi-scale features from medical images.
    """
    def __init__(self, pretrained=True):
        super().__init__()
        base = convnext_tiny(pretrained=pretrained)
        
        # Channel adapter: 1 (grayscale) → 3 (RGB for pretrained weights)
        self.channel_adapter = nn.Conv2d(1, 3, kernel_size=1, bias=False)
        nn.init.xavier_uniform_(self.channel_adapter.weight)
        
        # Extract ConvNeXt stages (4 resolution levels like CACO's ResNet)
        self.stage0 = base.features[0]                      # [B, 96, H/4, W/4]
        self.stage1 = nn.Sequential(*base.features[1:3])    # [B, 192, H/8, W/8]
        self.stage2 = nn.Sequential(*base.features[3:5])    # [B, 384, H/16, W/16]
        self.stage3 = nn.Sequential(*base.features[5:7])    # [B, 768, H/32, W/32]
        
        self.out_channels = 768  # Output channels from stage3
        
    def forward(self, x):
        """
        Args:
            x: [B, 1, H, W] grayscale images
        Returns:
            features: [B, 768, H/32, W/32] encoded features
        """
        x = self.channel_adapter(x)
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x


class ProjectionHead(nn.Module):  #This can also be improved upon. Currently the setup is a basic one.
    """
    MLP projection head.
    Projects encoder features to normalized embedding space.
    
    Architecture: Linear → BN → ReLU → Linear
    """
    def __init__(self, in_dim, hidden_dim=512, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] spatial features
        Returns:
            z: [B, out_dim] L2-normalized embeddings
        """
        # Global average pooling to get [B, C]
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        
        # Project to embedding space
        z = self.net(x)
        
        # L2 normalize (critical for contrastive learning)
        return F.normalize(z, dim=1)


class ContrastBank(nn.Module):
    """
    Learnable contrast prototypes for each modality.
    
    Stores 4 learnable vectors representing T1, T1CE, T2, FLAIR appearance.
    These are learned during training via alignment loss.
    At inference, we can query the bank for any modality without needing reference images!
    """
    def __init__(self, num_modalities=4, dim=128):
        super().__init__()
        # Learnable prototypes: [num_modalities, dim]
        self.prototypes = nn.Parameter(torch.randn(num_modalities, dim))
        
        # Initialize with normalized random vectors
        with torch.no_grad():
            self.prototypes.data = F.normalize(self.prototypes.data, dim=1)
    
    def forward(self, modality_ids):
        """
        Get contrast prototypes for given modality IDs.
        
        Args:
            modality_ids: [B] tensor of modality indices (0=T1, 1=T1CE, 2=T2, 3=FLAIR)
        Returns:
            prototypes: [B, dim] L2-normalized contrast prototypes
        """
        # Index into prototypes
        protos = self.prototypes[modality_ids]  # [B, dim]
        
        # L2 normalize (maintain on unit sphere)
        return F.normalize(protos, dim=1)
    
    def get_all_prototypes(self):
        """Get all 4 prototypes (for visualization/analysis)."""
        return F.normalize(self.prototypes, dim=1)

# ==============================================================================
# ADVERSARIAL AND DECORRELATION COMPONENTS
# ==============================================================================

class GradReverse(torch.autograd.Function):
    """Gradient Reversal Layer for adversarial modality classifier."""
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


class ModalityClassifier(nn.Module):
    """Small classifier to predict modality from anatomy embedding (used adversarially)."""
    def __init__(self, in_dim, num_modalities=4, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_modalities)
        )

    def forward(self, x):
        return self.net(x)


def decorrelation_loss(A, C):
    """Encourages anatomy and contrast embeddings to be uncorrelated."""
    A = A - A.mean(0, keepdim=True)
    C = C - C.mean(0, keepdim=True)
    N = A.size(0)
    cov = (A.T @ C) / (N - 1)
    return cov.pow(2).sum()

def prototype_orthogonality_loss(prototypes: torch.Tensor) -> torch.Tensor:
    """
    prototypes: [M, D] assumed normalized
    Returns a scalar penalizing off-diagonal cosine similarities.
    """
    # Cosine similarity matrix [M, M]
    sim = torch.matmul(prototypes, prototypes.t())
    # Zero out diagonal
    eye = torch.eye(sim.size(0), device=sim.device)
    off_diag = sim * (1.0 - eye)
    # Penalize squared off-diagonal similarity (mean)
    return off_diag.pow(2).mean()


class ContrastiveModel(nn.Module):
    """
    Main model combining encoder and dual projection heads + contrast bank.
    
     multi-key framework:
    - Key 0 (Anatomy): Invariant to all augmentations (modality changes)
    - Key 1 (Contrast): Variant to temporal/modality changes
    
    NEW: Added contrast bank to store learnable modality prototypes!
    """
    def __init__(
        self, 
        encoder_dim=768, 
        proj_dim=256,
        hidden_dim=512,
        num_modalities=4
    ):
        super().__init__()
        
        # Shared encoder (ConvNeXt)
        self.encoder = ConvNeXtEncoder(pretrained=True)
        
        # Two projection heads for two contrastive spaces
        self.anatomy_head = ProjectionHead(encoder_dim, hidden_dim, proj_dim)
        self.contrast_head = ProjectionHead(encoder_dim, hidden_dim, proj_dim)
        
        # Contrast bank storing learnable modality prototypes
        self.contrast_bank = ContrastBank(num_modalities=num_modalities, dim=proj_dim)

        # Add modality classifier (adversarial head)
        self.modality_clf = ModalityClassifier(proj_dim, num_modalities)

        
    def forward(self, images, modality_ids=None):
        """
        Args:
            images: [B, 1, H, W] batch of images
            modality_ids: [B] modality indices (optional, needed for bank)
        Returns:
            anatomy_emb: [B, proj_dim] anatomy embeddings
            contrast_emb: [B, proj_dim] contrast embeddings from images
            contrast_bank_emb: [B, proj_dim] contrast embeddings from bank (if modality_ids provided)
            features: [B, encoder_dim, H', W'] encoder features (for visualization)
        """
        # Extract features
        features = self.encoder(images)  # [B, 768, H/32, W/32]
        
        # Project to two embedding spaces
        anatomy_emb = self.anatomy_head(features)
        contrast_emb = self.contrast_head(features)
        
        # Get contrast from bank if modality_ids provided
        contrast_bank_emb = None
        if modality_ids is not None:
            contrast_bank_emb = self.contrast_bank(modality_ids)
        
        return anatomy_emb, contrast_emb, contrast_bank_emb, features


# ==============================================================================
# CONTRASTIVE LOSS
# ==============================================================================

def info_nce_loss(
    embeddings: torch.Tensor,
    positive_mask: torch.Tensor,
    temperature: float = 0.07
) -> torch.Tensor:
    """
    InfoNCE contrastive loss.
    
    Args:
        embeddings: [N, D] L2-normalized embeddings
        positive_mask: [N, N] binary mask where positive_mask[i,j]=1 if i,j are positives
        temperature: softmax temperature (default 0.07 from MoCo)
    
    Returns:
        loss: scalar contrastive loss
    """
    N = embeddings.shape[0]
    
    # Compute similarity matrix: [N, N]
    # Since embeddings are L2-normalized, this is cosine similarity
    sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature
    
    # Remove diagonal (self-similarity)
    positive_mask = positive_mask.clone()
    positive_mask.fill_diagonal_(0)
    
    # For numerical stability, subtract max before exp
    sim_matrix_max, _ = sim_matrix.max(dim=1, keepdim=True)
    sim_matrix = sim_matrix - sim_matrix_max.detach()
    
    # Compute exp(similarity) for all pairs
    exp_sim = torch.exp(sim_matrix)
    
    # Sum over negatives (all pairs except positives and self)
    # negative_mask = 1 - positive_mask
    # negative_mask.fill_diagonal_(0)
    # But we can just use sum of all exp_sim and subtract positives
    
    # Denominator: sum over all negatives
    # denominator = sum_j exp(sim(i,j)) where j != i
    denominator = exp_sim.sum(dim=1) - torch.diag(exp_sim)  # Exclude diagonal
    
    # Numerator: sum over positives
    # log_prob = log(exp(sim(i,pos)) / denominator)
    #          = sim(i,pos) - log(denominator)
    log_denominator = torch.log(denominator + 1e-8)
    
    # For each sample, compute loss over its positives
    num_positives = positive_mask.sum(dim=1)
    
    # Compute log probability for positive pairs
    # sim_matrix[i,j] where positive_mask[i,j] = 1
    log_prob = sim_matrix - log_denominator.unsqueeze(1)
    
    # Mask to keep only positive pairs and average
    loss_per_sample = -(log_prob * positive_mask).sum(dim=1) / (num_positives + 1e-8)
    
    # Filter out samples with no positives
    valid_samples = num_positives > 0
    if valid_samples.sum() == 0:
        return torch.tensor(0.0, device=embeddings.device)
    
    return loss_per_sample[valid_samples].mean()


def compute_contrastive_losses(
    anatomy_emb: torch.Tensor,
    contrast_emb: torch.Tensor,
    contrast_bank_emb: torch.Tensor,
    patient_ids: torch.Tensor,
    modality_ids: torch.Tensor,
    temperature: float = 0.07
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    
    1. Anatomy Loss: Pulls together embeddings from same patient, different modality
       → Learns anatomy features invariant to contrast/modality
    
    2. Contrast Loss: Pulls together embeddings from same modality, different patient
       → Learns contrast features invariant to anatomy/patient
    
    3. Bank Alignment Loss: Aligns bank prototypes with actual image contrasts
       → Makes bank prototypes learn meaningful modality representations
    
    Args:
        anatomy_emb: [N, D] anatomy embeddings
        contrast_emb: [N, D] contrast embeddings from images
        contrast_bank_emb: [N, D] contrast embeddings from bank
        patient_ids: [N] patient IDs
        modality_ids: [N] modality IDs
        temperature: contrastive temperature
    
    Returns:
        loss_anatomy: scalar anatomy contrastive loss
        loss_contrast: scalar contrast contrastive loss
        loss_bank_alignment: scalar bank alignment loss
    """
    N = anatomy_emb.shape[0]
    
    # Create masks for positive pairs
    # [N, N] boolean mask
    same_patient = (patient_ids.unsqueeze(0) == patient_ids.unsqueeze(1)).float()
    same_modality = (modality_ids.unsqueeze(0) == modality_ids.unsqueeze(1)).float()
    
    # Anatomy positives: same patient, different modality
    anatomy_positive_mask = same_patient * (1 - same_modality)
    
    # Contrast positives: same modality, different patient
    contrast_positive_mask = same_modality * (1 - same_patient)
    
    # Compute contrastive losses
    loss_anatomy = info_nce_loss(anatomy_emb, anatomy_positive_mask, temperature)
    loss_contrast = info_nce_loss(contrast_emb, contrast_positive_mask, temperature)
    
    # NEW: Bank alignment loss
    # Make bank prototypes match the contrast extracted from images
    # Use cosine similarity loss (since embeddings are L2-normalized)
    cosine_sim = F.cosine_similarity(contrast_bank_emb, contrast_emb.detach(), dim=1)
    loss_bank_alignment = 1 - cosine_sim.mean()  # Maximize similarity
    
    # Alternative: MSE loss (uncomment if preferred) or can use a weighted sum of both
    # loss_bank_alignment = F.mse_loss(contrast_bank_emb, contrast_emb.detach())
    
    return loss_anatomy, loss_contrast, loss_bank_alignment


# ==============================================================================
# TRAINING FUNCTIONS
# ==============================================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    anatomy_loss_sum = 0
    contrast_loss_sum = 0
    bank_loss_sum = 0  # NEW
    num_batches = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{num_epochs}")
    
    for batch in pbar:
        # Move data to device
        images = batch['images'].to(DEVICE, non_blocking=True)
        patient_ids = batch['patient_id'].to(DEVICE, non_blocking=True)
        modality_ids = batch['modality_id'].to(DEVICE, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast(dtype=torch.bfloat16):
            # Forward pass (NOW includes modality_ids for bank)
            anatomy_emb, contrast_emb, contrast_bank_emb, _ = model(images, modality_ids)
            
            # Compute contrastive losses (NOW includes bank alignment)
            loss_anatomy, loss_contrast, loss_bank = compute_contrastive_losses(
                anatomy_emb, contrast_emb, contrast_bank_emb,
                patient_ids, modality_ids,
                temperature=temperature
            )

            # 1️⃣ Adversarial modality classifier
            # Reverse gradients from anatomy embedding
            grl_lambda = min(1.0, epoch / 50)  # slowly ramp up adversarial strength
            rev_anat = grad_reverse(anatomy_emb, lambd=grl_lambda)
            logits_mod = model.modality_clf(rev_anat)
            loss_mod_adv = F.cross_entropy(logits_mod, modality_ids.long())

            # 2️⃣ Decorrelation between anatomy and contrast embeddings
            loss_decor = decorrelation_loss(anatomy_emb, contrast_emb)

            prototypes = model.contrast_bank.get_all_prototypes()  # [4, D]
            loss_proto_orth = prototype_orthogonality_loss(prototypes)
            
            # Total loss (NOW includes bank alignment)
            loss = (
                lambda_anatomy * loss_anatomy + 
                lambda_contrast * loss_contrast +
                lambda_bank_alignment * loss_bank +
                lambda_modality_adv * loss_mod_adv +
                lambda_decor * loss_decor +
                lambda_proto_orth * loss_proto_orth
            )
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        with torch.no_grad():
            model.contrast_bank.prototypes.data = F.normalize(model.contrast_bank.prototypes.data, dim=1)
        scaler.update()
        
        total_loss += loss.item()
        anatomy_loss_sum += loss_anatomy.item()
        contrast_loss_sum += loss_contrast.item()
        bank_loss_sum += loss_bank.item()
        adv_loss_sum = locals().get("adv_loss_sum", 0) + loss_mod_adv.item()
        decor_loss_sum = locals().get("decor_loss_sum", 0) + loss_decor.item()
        proto_oth_loss_sum = loss_proto_orth.item()
        num_batches += 1

        pbar.set_postfix({
            "L_total": f"{loss.item():.4f}",
            "L_anat": f"{loss_anatomy.item():.4f}",
            "L_cont": f"{loss_contrast.item():.4f}",
            "L_bank": f"{loss_bank.item():.4f}",
            "L_adv": f"{loss_mod_adv.item():.4f}",
            "L_decor": f"{loss_decor.item():.4f}",
            "L_orth":f"{loss_proto_orth.item():.4f}"

        })

    
    return {
        'total': total_loss / num_batches,
        'anatomy': anatomy_loss_sum / num_batches,
        'contrast': contrast_loss_sum / num_batches,
        'bank_alignment': bank_loss_sum / num_batches,
        'adv_modality': adv_loss_sum / num_batches,
        'decorrelation': decor_loss_sum / num_batches,
        'ortho':proto_oth_loss_sum/num_batches
    }

@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()
    
    total_loss = 0
    anatomy_loss_sum = 0
    contrast_loss_sum = 0
    bank_loss_sum = 0  # NEW
    num_batches = 0
    
    for batch in tqdm(loader, desc="Validating"):
        images = batch['images'].to(DEVICE, non_blocking=True)
        patient_ids = batch['patient_id'].to(DEVICE, non_blocking=True)
        modality_ids = batch['modality_id'].to(DEVICE, non_blocking=True)
        
        with autocast(dtype=torch.bfloat16):
            anatomy_emb, contrast_emb, contrast_bank_emb, _ = model(images, modality_ids)

            # Optionally measure modality leakage in anatomy embeddings
            logits_val = model.modality_clf(anatomy_emb)
            val_acc = (logits_val.argmax(dim=1) == modality_ids).float().mean().item()
            print(f"Anatomy->Modality Accuracy: {val_acc:.3f}")

            
            loss_anatomy, loss_contrast, loss_bank = compute_contrastive_losses(
                anatomy_emb, contrast_emb, contrast_bank_emb,
                patient_ids, modality_ids,
                temperature=temperature
            )
            
            loss = (
                lambda_anatomy * loss_anatomy + 
                lambda_contrast * loss_contrast +
                lambda_bank_alignment * loss_bank
            )
        
        total_loss += loss.item()
        anatomy_loss_sum += loss_anatomy.item()
        contrast_loss_sum += loss_contrast.item()
        bank_loss_sum += loss_bank.item()  # NEW
        num_batches += 1
    
    return {
        'total': total_loss / num_batches,
        'anatomy': anatomy_loss_sum / num_batches,
        'contrast': contrast_loss_sum / num_batches,
        'bank_alignment': bank_loss_sum / num_batches  # NEW
    }


# ==============================================================================
# VISUALIZATION UTILITIES
# ==============================================================================

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import numpy as np

@torch.no_grad()
def visualize_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    epoch: int,
    save_dir: str,
    max_samples: int = 512,
    use_umap: bool = False
):
    """
    Visualizes anatomy and contrast embedding spaces, along with contrast bank prototypes.
    Saves t-SNE plots to `save_dir`.

    Args:
        model: trained model
        loader: DataLoader (e.g., val_loader)
        device: 'cuda' or 'cpu'
        epoch: current epoch number
        save_dir: directory to save plots
        max_samples: max number of samples to visualize
        use_umap: if True, use UMAP instead of t-SNE
    """
    import umap.umap_ as umap
 # Optional dependency

    model.eval()

    all_anat, all_cont, all_mod, all_pat = [], [], [], []
    for batch in loader:
        images = batch["images"].to(device)
        modality_ids = batch["modality_id"].to(device)
        patient_ids = batch["patient_id"].to(device)

        anat, cont, _, _ = model(images, modality_ids)
        all_anat.append(anat.cpu())
        all_cont.append(cont.cpu())
        all_mod.append(modality_ids.cpu())
        all_pat.append(patient_ids.cpu())

        if len(all_anat) * images.size(0) >= max_samples:
            break

    all_anat = torch.cat(all_anat)[:max_samples]
    all_cont = torch.cat(all_cont)[:max_samples]
    all_mod = torch.cat(all_mod)[:max_samples]

    # Get contrast bank prototypes
    bank = model.contrast_bank.get_all_prototypes().cpu()

    # Dimensionality reduction
    # reducer = (
    #     umap.UMAP(random_state=42)
    #     if use_umap
    #     else TSNE(n_components=2, perplexity=30, random_state=42)
    # )
    # anat_2d = reducer.fit_transform(all_anat)
    # cont_2d = reducer.fit_transform(all_cont)
    # bank_2d = reducer.fit_transform(bank)

    # ----- fit reducer jointly on contrast embeddings + prototypes -----
    import numpy as np

    all_cont_np = all_cont.numpy()
    bank_np = bank.numpy()
    combined = np.vstack([all_cont_np, bank_np])  # [N + 4, D]

    if use_umap:
        reducer = umap.UMAP(random_state=42)
        combined_2d = reducer.fit_transform(combined)
    else:
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
        combined_2d = reducer.fit_transform(combined)

    cont_2d = combined_2d[: all_cont_np.shape[0], :]
    bank_2d = combined_2d[all_cont_np.shape[0] :, :]

    # Optionally also reduce anatomy separately (doesn't need to share frame)
    if use_umap:
        anat_reducer = umap.UMAP(random_state=42)
        anat_2d = anat_reducer.fit_transform(all_anat)
    else:
        anat_reducer = TSNE(n_components=2, perplexity=30, random_state=42)
        anat_2d = anat_reducer.fit_transform(all_anat)
    # -------------------------------------------------------------------------


    modalities = ["T1", "T1CE", "T2", "FLAIR"]
    palette = sns.color_palette("tab10", len(modalities))

    # Plot anatomy space (should mix modalities)
    plt.figure(figsize=(7, 6))
    for m in range(len(modalities)):
        idx = (all_mod == m)
        plt.scatter(
            anat_2d[idx, 0],
            anat_2d[idx, 1],
            s=8,
            label=modalities[m],
            alpha=0.7,
            color=palette[m],
        )
    plt.legend()
    plt.title(f"Anatomy Embedding Space (Epoch {epoch})")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"anatomy_space_epoch{epoch:03d}.png"))
    plt.close()

    # Plot contrast space (should cluster by modality)
    plt.figure(figsize=(7, 6))
    for m in range(len(modalities)):
        idx = (all_mod == m)
        plt.scatter(
            cont_2d[idx, 0],
            cont_2d[idx, 1],
            s=8,
            label=modalities[m],
            alpha=0.7,
            color=palette[m],
        )

    # Plot prototypes (contrast bank)
    plt.scatter(
        bank_2d[:, 0],
        bank_2d[:, 1],
        s=250,
        marker="*",
        c=palette,
        edgecolor="k",
        linewidth=1.5,
        label="Bank Prototypes",
    )

    plt.legend()
    plt.title(f"Contrast Embedding Space (Epoch {epoch})")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"contrast_space_epoch{epoch:03d}.png"))
    plt.close()

    print(f"Saved embedding visualizations for epoch {epoch} → {save_dir}")

    # Simple prototype diagnostics
    protos = model.contrast_bank.get_all_prototypes().to(device)
    contrast_embeddings = all_cont.to(device)
    modality_ids = all_mod.to(device)

    sims = torch.matmul(contrast_embeddings, protos.t())
    for m in range(len(modalities)):
        mask = (modality_ids == m)
        if mask.any():
            print(f"{modalities[m]} mean sim to its proto: {sims[mask, m].mean().item():.3f}")
    proto_sims = torch.matmul(protos, protos.t())
    print("Inter-prototype cosine similarities:\n", proto_sims.cpu().numpy())




def save_checkpoint(state: Dict, filepath: str):
    """Save model checkpoint."""
    torch.save(state, filepath)
    print(f"Saved checkpoint: {filepath}")


# ==============================================================================
# MAIN TRAINING LOOP
# ==============================================================================

if __name__ == "__main__":
    print(f"\n{'='*80}")
    print(f"{'='*80}\n")
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_contrastive_dataloaders(
        train_root=train_data_root,
        val_root=val_data_root,
        modalities=modalities,
        num_patients_per_batch=num_patients_per_batch,
        num_slices_per_modality=num_slices,
        target_size=target_size,
        num_workers=8,
        apply_mask=True
    )
    
    # Create model
    print("Building model...")
    model = ContrastiveModel(
        encoder_dim=768,
        proj_dim=emb_dim,
        hidden_dim=512
    ).to(DEVICE)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {num_params:,}")
    
    # Create optimizer (AdamW like in modern contrastive learning)
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler (cosine annealing like CACO)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=lr * 0.01
    )
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    print(f"\n{'='*80}")
    print("TRAINING CONFIGURATION")
    print(f"{'='*80}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {lr}")
    print(f"Weight decay: {weight_decay}")
    print(f"Patients per batch: {num_patients_per_batch}")
    print(f"Effective batch size: {num_patients_per_batch * len(modalities)} images")
    print(f"Temperature: {temperature}")
    print(f"Device: {DEVICE}")
    print(f"{'='*80}\n")
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch}/{num_epochs}")
        print(f"{'='*80}")
        
        # Train
        train_losses = train_epoch(model, train_loader, optimizer, scaler, epoch)
        
        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"\nTraining Summary:")
        print(f"   Total Loss: {train_losses['total']:.4f}")
        print(f"   Anatomy Loss: {train_losses['anatomy']:.4f}")
        print(f"   Contrast Loss: {train_losses['contrast']:.4f}")
        print(f"   Bank Alignment Loss: {train_losses['bank_alignment']:.4f}")  # NEW
        print(f"   Learning Rate: {current_lr:.2e}")
        print(f"   Adversarial Modality Loss: {train_losses['adv_modality']:.4f}")
        print(f"   Decorrelation Loss: {train_losses['decorrelation']:.4f}")
        
        # Validate every N epochs
        if epoch % 10 == 0 or epoch == 1:
            val_losses = validate(model, val_loader)
            
            print(f"\nValidation Summary:")
            print(f"   Total Loss: {val_losses['total']:.4f}")
            print(f"   Anatomy Loss: {val_losses['anatomy']:.4f}")
            print(f"   Contrast Loss: {val_losses['contrast']:.4f}")
            print(f"   Bank Alignment Loss: {val_losses['bank_alignment']:.4f}")  # NEW
            
            # Save best model
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'best_val_loss': best_val_loss,
                    'config': {
                        'encoder_dim': 768,
                        'proj_dim': emb_dim,
                        'temperature': temperature,
                        'modalities': modalities,
                    }
                }, os.path.join(logdir, "best_model.pth"))

        if epoch % 20 == 0 or epoch == 1:
            print(f"Visualizing embedding spaces...")
            visualize_embeddings(
                model=model,
                loader=val_loader,
                device=DEVICE,
                epoch=epoch,
                save_dir=logdir,
                max_samples=512,
                use_umap=True  # set True if UMAP installed
            )
        
        # Save checkpoint every 20 epochs
        if epoch % 20 == 0:
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'train_losses': train_losses,
                'config': {
                    'encoder_dim': 768,
                    'proj_dim': emb_dim,
                    'temperature': temperature,
                    'modalities': modalities,
                }
            }, os.path.join(logdir, f"checkpoint_epoch{epoch:03d}.pth"))
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETED!")
    print(f"{'='*80}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {logdir}")
    print(f"\nNext steps:")
    print("   1. Load best_model.pth")
    print("   2. Extract anatomy features: model.encoder + model.anatomy_head")
    print("   3. Use for downstream tasks (segmentation, classification, etc.)")
    print(f"{'='*80}\n")