import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import cv2
from model import ViT_cifar10
import os

# ---- 1. Attention Rollout ----
def compute_attention_rollout(attn_weights, discard_ratio=0.0):
    num_layers = len(attn_weights)
    batch_size = attn_weights[0].shape[0]
    num_tokens = attn_weights[0].shape[-1]
    result = torch.eye(num_tokens).unsqueeze(0).repeat(batch_size, 1, 1).to(attn_weights[0].device)

    for attn in attn_weights:
        attn_heads_fused = attn.mean(dim=1)
        if discard_ratio > 0:
            flat = attn_heads_fused.view(batch_size, -1)
            _, indices = flat.topk(int(flat.shape[-1] * (1 - discard_ratio)), dim=-1)
            mask = torch.zeros_like(flat)
            mask.scatter_(1, indices, 1.0)
            attn_heads_fused = (attn_heads_fused.view(batch_size, -1) * mask).view_as(attn_heads_fused)

        attn_heads_fused = attn_heads_fused + torch.eye(num_tokens).to(attn_heads_fused.device)
        attn_heads_fused = attn_heads_fused / attn_heads_fused.sum(dim=-1, keepdim=True)
        result = torch.bmm(attn_heads_fused, result)

    rollout = result[:, 0, 1:]  # CLS token attention to patches
    return rollout


# ---- 2. Save individual visualizations ----
# ---- 2. Save individual visualizations ----
def save_individual_attention_images(images, rollouts, patch_size, class_names, preds, labels, save_dir="./outputs"):
    os.makedirs(save_dir, exist_ok=True)
    num_images = len(images)

    for i in range(num_images):
        image_tensor = images[i]
        rollout = rollouts[i].detach().cpu().numpy()

        # Convert tensor to numpy image
        image = image_tensor.permute(1, 2, 0).cpu().numpy()
        image = (image - image.min()) / (image.max() - image.min())

        # Attention map
        num_patches = rollout.shape[-1]
        grid_size = int(np.sqrt(num_patches))
        attn_map = rollout.reshape(grid_size, grid_size)
        attn_map = cv2.resize(attn_map, (image.shape[1], image.shape[0]))
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())

        # Plot
        plt.figure(figsize=(6, 3))

        # Left: original image with true class
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"True: {class_names[labels[i]]}")

        # Right: overlayed attention with predicted class
        plt.subplot(1, 2, 2)
        plt.imshow(image)
        plt.imshow(attn_map, cmap='jet', alpha=0.5)
        plt.axis('off')
        plt.title(f"Predicted: {class_names[preds[i]]}")

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"sample_{i+1}_true_{class_names[labels[i]]}_pred_{class_names[preds[i]]}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"[✅] Saved visualization: {save_path}")


# ---- 3. CIFAR-10 Loader ----
transform = transforms.Compose([transforms.ToTensor()])
valset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=5, shuffle=True)
class_names = valset.classes

# ---- 4. Get 5 Images ----
images, labels = next(iter(valloader))
images = images[:5]
labels = labels[:5]

# ---- 5. Model ----
device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = {
    "image_size": 32,
    "patch_size": 4,
    "num_channels": 3,
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_attention_heads": 8,
    "num_hidden_layers": 8,
    "hidden_dropout_prob": 0.1,
    "num_classes": 10,
    "bias": True,
    "patchify_method": "conv"
}

model = ViT_cifar10(config).to(device)

# ✅ Load pretrained weights if available
checkpoint_path = "/home/cha0s/Downloads/vit_cifar10_12_75_8.pth"
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"[✅] Loaded model weights from {checkpoint_path}")
else:
    print("[⚠️] Warning: No pretrained weights found, using random initialization.")

model.eval()

# ---- 6. Forward Pass ----
with torch.no_grad():
    logits, attn_weights = model(images.to(device))
    preds = logits.argmax(dim=1).cpu().numpy()

# ---- 7. Compute Rollouts ----
rollout = compute_attention_rollout(attn_weights)

# ---- 8. Save each visualization ----
save_individual_attention_images(images, rollout, config['patch_size'], class_names, preds, labels, save_dir="./outputs")
