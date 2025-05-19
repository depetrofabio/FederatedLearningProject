import torch.nn as nn
import torch

def debug_model(model: nn.Module, model_name: str = "Model"):
    """
    Prints debugging information about a PyTorch model.

    Information includes:
    - Overall device of the first parameter (indicative of model's primary device).
    - For each named parameter:
        - Full parameter name.
        - Device of the parameter.
        - Whether the parameter requires gradients (is frozen or not).
        - Inferred block index if the name matches a ViT-like structure.
    """
    print(f"\n--- Debugging {model_name} ---")

    # Check overall model device (based on the first parameter)
    try:
        first_param_device = next(model.parameters()).device
        print(f"{model_name} is primarily on device: {first_param_device}")
    except StopIteration:
        print(f"{model_name} has no parameters.")
        return

    print("\nParameter Details (Name | Device | Requires Grad? | Inferred Block):")
    for name, param in model.named_parameters():
        device = param.device
        requires_grad = param.requires_grad

        block_info = "N/A"
        # Try to infer block index for ViT-like models
        if "blocks." in name:
            try:
                # e.g., name = "blocks.0.attn.qkv.weight"
                block_idx_str = name.split("blocks.")[1].split(".")[0]
                if block_idx_str.isdigit():
                    block_info = f"Block {block_idx_str}"
            except IndexError:
                block_info = "Block (parse error)"

        print(f"- {name:<50} | {str(device):<10} | {str(requires_grad):<15} | {block_info}")

    # You can add more specific checks here, e.g., for model mode (train/eval)
    print(f"{model_name} is in {'training' if model.training else 'evaluation'} mode.")
    print(f"--- End Debugging {model_name} ---\n")

def load_backbone():
    return torch.hub.load('facebookresearch/dino:main', 'dino_vits16')

def move_to_cuda(model):
    avaiability = torch.cuda.is_available()
    if avaiability:
        device = torch.device("cuda")
        print("moving model to cuda")
    else:
        device = torch.device("cpu")
        print("cuda not available")
    model.to(device)
    return avaiability

# ========== HEAD ONLY =========== #
class DinoOnlyHead(nn.Module):

    def __init__(self, num_classes=100, hidden_dim=256, drop=0.5):
        super().__init__()
        backbone = load_backbone()
        backbone.eval()

        # froze the backbone
        for p in backbone.parameters():
            p.requires_grad = False

        self.backbone = backbone

        embed_dim = backbone.embed_dim  # 384 for ViT-S/16

        # attach the head
        self.head = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # get CLS token from the frozen backbone
        with torch.no_grad():
            cls = self.backbone.forward_features(x)   # -> (B, embed_dim)
        return self.head(cls)
    
    def debug(self):
        debug_model(self)

    def to_cuda(self):
        move_to_cuda(self)


# ========== 3 UNFROZEN BLOCKS =========== #
"""
pos_drop is likely an nn.Dropout layer associated with positional embeddings. nn.Dropout layers themselves don't have learnable parameters
that are updated during backpropagation (only a dropout rate, which is a hyperparameter). So, iterating through parameters() of a standard
dropout layer might yield an empty iterator or no parameters that gradients flow through. The main control over dropout is its train() or
eval() mode. However, if pos_drop were a custom module with learnable parameters, this would freeze them. This line is unlikely to cause
issues but might not have a significant effect if pos_drop is a standard nn.Dropout.

for p in backbone.pos_drop.parameters():     # da verificare, non dovrebbero esserci parametri trainabili nei drop out layers
    p.requires_grad = False
"""
class HeadAnd3Blocks(nn.Module):

    def __init__(self, num_classes=100, hidden_dim=256, drop=0.5): # hidden_dim = dimensione del layer della nn
        super().__init__()
        backbone = load_backbone()

        # Freeze patch embedding and dropout
        for p in backbone.patch_embed.parameters():  # embedding vectors sono rappresentazioni numeriche dei dati
            p.requires_grad = False

        # Freeze first 9 blocks
        for block in backbone.blocks[0:9]:
            block.eval()
            for param in block.parameters():
                param.requires_grad = False

        # Unfreeze remaining blocks (if needed)
        for block in backbone.blocks[9:]:
            block.train()
            for param in block.parameters():
                param.requires_grad = True

        backbone.pos_embed.requires_grad = False
        backbone.cls_token.requires_grad = False

        self.backbone = backbone
        embed_dim = backbone.embed_dim  # 384 for ViT-S/16
        self.head = nn.Sequential(
            # nn.Dropout(drop),                   # solitamnete non si fa il dropout prima dell'input layer, da capire
            nn.Linear(embed_dim, hidden_dim),     # from 384 to 256
            nn.ReLU(inplace=True),                # capire meglio inplace
            nn.Dropout(drop),
            nn.Linear(hidden_dim, num_classes)    # from 256 to 100
        )

    def forward(self, x):
        feats = self.backbone.get_intermediate_layers(x, n=1)[0] # take the output features from DiNo's backbone
        cls = feats[:, 0]                                        #
        return self.head(cls)
    
    def debug(self):
        debug_model(self)
    
    def to_cuda(self):
        move_to_cuda(self)
