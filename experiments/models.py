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
        - Mode of the module containing the parameter (Train/Eval) if applicable.
    """
    print(f"\n--- Debugging {model_name} ---")
    try:
        # Check if the model has any parameters before trying to access the first one
        if not list(model.parameters()):
            print(f"{model_name} has no parameters.")
            # Optionally, print overall model mode if it has no parameters but is a module
            if hasattr(model, 'training'):
                 model_mode_overall = "Train" if model.training else "Eval"
                 print(f"{model_name} overall mode: {model_mode_overall}")
            print(f"--- End Debugging {model_name} ---\n")
            return
        first_param_device = next(model.parameters()).device
        print(f"{model_name} is primarily on device: {first_param_device}")
        if hasattr(model, 'training'): # Also print overall model mode
            model_mode_overall = "Train" if model.training else "Eval"
            print(f"{model_name} overall mode: {model_mode_overall}")
    except StopIteration:
        # This StopIteration should ideally be caught by the check above,
        # but it's a fallback.
        print(f"{model_name} has no parameters (StopIteration).")
        if hasattr(model, 'training'):
             model_mode_overall = "Train" if model.training else "Eval"
             print(f"{model_name} overall mode: {model_mode_overall}")
        print(f"--- End Debugging {model_name} ---\n")
        return
    # Modified header for the new column
    print("\nParameter Details (Name | Device | Requires Grad? | Inferred Block | Module Mode):") # Header changed to "Module Mode"
    for name, param in model.named_parameters():
        device = param.device
        requires_grad = param.requires_grad
        block_info = "N/A"
        module_mode_str = "N/A"  # Renamed from block_mode_str for clarity
        if "blocks." in name:
            try:
                # e.g., name = "backbone.blocks.0.attn.qkv.weight" or "blocks.0.attn.qkv.weight"
                name_parts = name.split("blocks.")
                block_idx_str = name_parts[1].split(".")[0]
                if block_idx_str.isdigit():
                    block_info = f"Block {block_idx_str}"
                    try:
                        parent_module_of_blocks = model
                        # Navigate to the parent module of 'blocks' if a path is prefixed
                        # e.g., if name_parts[0] is "backbone.", navigate to model.backbone
                        if name_parts[0]:
                            for part in name_parts[0].rstrip('.').split('.'):
                                parent_module_of_blocks = getattr(parent_module_of_blocks, part)
                        # Now, parent_module_of_blocks is the module that should contain the 'blocks' attribute
                        actual_block_module = parent_module_of_blocks.blocks[int(block_idx_str)]
                        if hasattr(actual_block_module, 'training'):
                             module_mode_str = "Train" if actual_block_module.training else "Eval"
                    except Exception:
                        pass # Keep module_mode_str as "N/A" if mode cannot be determined
            except IndexError:
                block_info = "Block (parse error)"
        else:
            # Logic for parameters not in 'blocks.X' (e.g., head, backbone.norm)
            try:
                # Get all parts of the name except the parameter name itself to find the parent module
                module_path_parts = name.split('.')[:-1]
                current_parent_module = model # Start from the top-level model
                if module_path_parts: # If the parameter is nested (e.g., "head.0.weight")
                    for part_name in module_path_parts:
                        if part_name.isdigit() and hasattr(current_parent_module, '__getitem__') and not isinstance(current_parent_module, dict):
                            # Access elements of nn.ModuleList or nn.Sequential by index
                            current_parent_module = current_parent_module[int(part_name)]
                        else:
                            # Access submodules by attribute name
                            current_parent_module = getattr(current_parent_module, part_name)
                # After the loop, current_parent_module is the direct parent of the parameter,
                # or the model itself if the parameter is directly attached to the model.
                if hasattr(current_parent_module, 'training'):
                    module_mode_str = "Train" if current_parent_module.training else "Eval"
            except Exception:
                pass # Keep module_mode_str "N/A" if any error occurs during module path traversal
        # Modified print statement to include module_mode_str and adjust spacing for block_info
        print(f"- {name:<50} | {str(device):<10} | {str(requires_grad):<15} | {block_info:<15} | {module_mode_str}")
    print(f"--- End Debugging {model_name} ---\n")

def load_backbone():
    return torch.hub.load('facebookresearch/dino:main', 'dino_vits16')

def move_to_cuda(model):
    availability = torch.cuda.is_available()
    if availability:
        device = torch.device("cuda")
        print("moving model to cuda")
    else:
        device = torch.device("cpu")
        print("cuda not available")
    model.to(device)
    return availability

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
            # nn.Dropout(drop),                   # solitamnete non si fa il dropout prima dell'input layer, da capire
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, num_classes)
        )
        self.head.train()
    
    def train(self, mode: bool = True):
        """Override the train method to ensure frozen parts stay in eval mode."""
        super().train(mode) # This will set self.training and call train(mode) on all children
        if mode:
            self.backbone.eval()       
        return self

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
        for p in backbone.patch_embed.parameters():  # embedding vectors sono le rappresentazioni numeriche dei dati
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
        self.head.train()
    
    def train(self, mode: bool = True):
        """Override the train method to ensure frozen parts stay in eval mode."""
        super().train(mode) # This will set self.training and call train(mode) on all children
        if mode:
            # If the model is being put into training mode,
            # explicitly set the designated frozen blocks back to eval mode.
            for i, block in enumerate(self.backbone.blocks):
                if i < 9:           # take blocks from 0 to 9 and set them in eval again
                    block.eval()        
        return self

    def forward(self, x):
        feats = self.backbone.get_intermediate_layers(x, n=1)[0] # take the output features from DiNo's backbone
        cls = feats[:, 0]                                        #
        return self.head(cls)
    
    def debug(self):
        debug_model(self)
    
    def to_cuda(self):
        move_to_cuda(self)


class FlexibleDino(nn.Module):
    def __init__(self, num_classes=100, hidden_dim=256, drop=0.5, num_layers_to_freeze=0):
        super().__init__()
        backbone = load_backbone()
        self.num_layers_frozen = num_layers_to_freeze

        if (num_layers_to_freeze == 0): # se passiamo 0 come numero di blocchi da freezzare non freezziamo nemmeno questi 2 layers
            backbone.pos_embed.requires_grad = True
            backbone.cls_token.requires_grad = True
            for p in backbone.patch_embed.parameters():  # embedding vectors sono le rappresentazioni numeriche dei dati
                p.requires_grad = True
        else:
            for p in backbone.patch_embed.parameters():  # embedding vectors sono le rappresentazioni numeriche dei dati
                p.requires_grad = False
            backbone.pos_embed.requires_grad = False
            backbone.cls_token.requires_grad = False

        # Freeze first num_layers_to_freeze blocks
        for block in backbone.blocks[0:num_layers_to_freeze]:
            block.eval()
            for param in block.parameters():
                param.requires_grad = False

        # Unfreeze remaining blocks (if needed)
        for block in backbone.blocks[num_layers_to_freeze:]:
            block.train()
            for param in block.parameters():
                param.requires_grad = True

        self.backbone = backbone
        embed_dim = backbone.embed_dim  # 384 for ViT-S/16
        self.head = nn.Sequential(
            # nn.Dropout(drop),                   # solitamnete non si fa il dropout prima dell'input layer, da capire
            nn.Linear(embed_dim, hidden_dim),     # from 384 to 256
            nn.ReLU(inplace=True),                # capire meglio inplace
            nn.Dropout(drop),
            nn.Linear(hidden_dim, num_classes)    # from 256 to 100
        )
        self.head.train()
    
    # @Override
    def train(self, mode: bool = True):
        """Override the train method to ensure frozen parts stay in eval mode."""
        super().train(mode) # This will set self.training and call train(mode) on all children
        if mode:
            # If the model is being put into training mode,
            # explicitly set the designated frozen blocks back to eval mode.
            for i, block in enumerate(self.backbone.blocks):
                if i < self.num_layers_frozen:           # take blocks from 0 to 9 and set them in eval again
                    block.eval()        
        return self

    def forward(self, x):
        feats = self.backbone.get_intermediate_layers(x, n=1)[0] # take the output features from DiNo's backbone
        cls = feats[:, 0]                                        #
        return self.head(cls)
    
    def debug(self):
        debug_model(self)
    
    def to_cuda(self):
        move_to_cuda(self)

    def freeze(self, num_blocks): # num_blocks = numero di blocchi che vuoi freezzare (da 0 a 12) 
        # es. se num_blocks = 4 allora i blocchi 0, 1, 2, 3 saranno freezzati
        num_total_blocks = len(self.backbone.blocks)

        if num_blocks > num_total_blocks:
            print(f"Warning: Requested to freeze {num_blocks} blocks, but backbone only has {num_total_blocks}")
            exit(0)
        elif num_blocks < 0:
            print(f"Warning: Requested to freeze {num_blocks} blocks.")
            exit(0)
        
        self.num_layers_frozen = num_blocks

        if (num_blocks == 0): # se passiamo 0 come numero di blocchi da freezzare non freezziamo nemmeno questi 2 layers
            self.backbone.pos_embed.requires_grad = True
            self.backbone.cls_token.requires_grad = True
            for p in self.backbone.patch_embed.parameters():  # embedding vectors sono le rappresentazioni numeriche dei dati
                p.requires_grad = True
        else:
            for p in self.backbone.patch_embed.parameters():  # embedding vectors sono le rappresentazioni numeriche dei dati
                p.requires_grad = False
            self.backbone.pos_embed.requires_grad = False
            self.backbone.cls_token.requires_grad = False

        for i in range(num_blocks):
            block = self.backbone.blocks[i]
            block.eval()
            for param in block.parameters():
                param.requires_grad = False
        
        for i in range(num_blocks, num_total_blocks):
            block = self.backbone.blocks[i]
            block.train()
            for param in block.parameters():
                param.requires_grad = True


    # la unfreeze fa la stessa cosa della unfreeze, cambia solo il parametro in input, si può cancellare
    def unfreeze(self, num_blocks): # num_blocks = numero di blocchi che vuoi unfreezzare 
        # es. se num_blocks = 3 allora i blocchi unfreezzati saranno il 9, il 10 e l'11
        num_total_blocks = len(self.backbone.blocks)

        if (num_blocks == 12): 
            self.backbone.pos_embed.requires_grad = True
            self.backbone.cls_token.requires_grad = True
            for p in self.backbone.patch_embed.parameters():  # embedding vectors sono le rappresentazioni numeriche dei dati
                p.requires_grad = True
        else:
            for p in self.backbone.patch_embed.parameters():  # embedding vectors sono le rappresentazioni numeriche dei dati
                p.requires_grad = False
            self.backbone.pos_embed.requires_grad = False
            self.backbone.cls_token.requires_grad = False

        if num_blocks > num_total_blocks:
            print(f"Warning: Requested to unfreeze {num_blocks} blocks, but backbone only has {num_total_blocks}.")
            exit(0)
        elif num_blocks < 0:
            print(f"Warning: Requested to unfreeze {num_blocks} blocks.")
            exit(0)
        
        self.num_layers_frozen = num_total_blocks - num_blocks
        start_index = num_total_blocks - num_blocks # es. se voglio defreezare 5 layer -> 12-5 = 7 -> 7-8-9-10-11 unfreezzati
        for i in range(start_index, num_total_blocks): 
            block = self.backbone.blocks[i]
            block.train() 
            for param in block.parameters():
                param.requires_grad = True

        for i in range(num_total_blocks - num_blocks):
            block = self.backbone.blocks[i]
            block.eval()
            for param in block.parameters():
                param.requires_grad = False


class LinearFlexibleDino(nn.Module):
    def __init__(self, num_classes=100, hidden_dim=256, drop=0.5, num_layers_to_freeze=0):
        super().__init__()
        backbone = load_backbone()
        self.num_layers_frozen = num_layers_to_freeze

        if (num_layers_to_freeze == 0): # se passiamo 0 come numero di blocchi da freezzare non freezziamo nemmeno questi 2 layers
            backbone.pos_embed.requires_grad = True
            backbone.cls_token.requires_grad = True
            for p in backbone.patch_embed.parameters():  # embedding vectors sono le rappresentazioni numeriche dei dati
                p.requires_grad = True
        else:
            for p in backbone.patch_embed.parameters():  # embedding vectors sono le rappresentazioni numeriche dei dati
                p.requires_grad = False
            backbone.pos_embed.requires_grad = False
            backbone.cls_token.requires_grad = False

        # Freeze first num_layers_to_freeze blocks
        for block in backbone.blocks[0:num_layers_to_freeze]:
            block.eval()
            for param in block.parameters():
                param.requires_grad = False

        # Unfreeze remaining blocks (if needed)
        for block in backbone.blocks[num_layers_to_freeze:]:
            block.train()
            for param in block.parameters():
                param.requires_grad = True

        self.backbone = backbone
        embed_dim = backbone.embed_dim  # 384 for ViT-S/16
        self.head = nn.Linear(384,100)
        self.head.train()
    
    # @Override
    def train(self, mode: bool = True):
        """Override the train method to ensure frozen parts stay in eval mode."""
        super().train(mode) # This will set self.training and call train(mode) on all children
        if mode:
            # If the model is being put into training mode,
            # explicitly set the designated frozen blocks back to eval mode.
            for i, block in enumerate(self.backbone.blocks):
                if i < self.num_layers_frozen:           # take blocks from 0 to 9 and set them in eval again
                    block.eval()        
        return self

    def forward(self, x):
        feats = self.backbone.get_intermediate_layers(x, n=1)[0] # take the output features from DiNo's backbone
        cls = feats[:, 0]                                        #
        return self.head(cls)
    
    def debug(self):
        debug_model(self)
    
    def to_cuda(self):
        move_to_cuda(self)

    def freeze(self, num_blocks): # num_blocks = numero di blocchi che vuoi freezzare (da 0 a 12) 
        # es. se num_blocks = 4 allora i blocchi 0, 1, 2, 3 saranno freezzati
        num_total_blocks = len(self.backbone.blocks)

        if num_blocks > num_total_blocks:
            print(f"Warning: Requested to freeze {num_blocks} blocks, but backbone only has {num_total_blocks}")
            exit(0)
        elif num_blocks < 0:
            print(f"Warning: Requested to freeze {num_blocks} blocks.")
            exit(0)
        
        self.num_layers_frozen = num_blocks

        if (num_blocks == 0): # se passiamo 0 come numero di blocchi da freezzare non freezziamo nemmeno questi 2 layers
            self.backbone.pos_embed.requires_grad = True
            self.backbone.cls_token.requires_grad = True
            for p in self.backbone.patch_embed.parameters():  # embedding vectors sono le rappresentazioni numeriche dei dati
                p.requires_grad = True
        else:
            for p in self.backbone.patch_embed.parameters():  # embedding vectors sono le rappresentazioni numeriche dei dati
                p.requires_grad = False
            self.backbone.pos_embed.requires_grad = False
            self.backbone.cls_token.requires_grad = False

        for i in range(num_blocks):
            block = self.backbone.blocks[i]
            block.eval()
            for param in block.parameters():
                param.requires_grad = False
        
        for i in range(num_blocks, num_total_blocks):
            block = self.backbone.blocks[i]
            block.train()
            for param in block.parameters():
                param.requires_grad = True


    # la unfreeze fa la stessa cosa della unfreeze, cambia solo il parametro in input, si può cancellare
    def unfreeze(self, num_blocks): # num_blocks = numero di blocchi che vuoi unfreezzare 
        # es. se num_blocks = 3 allora i blocchi unfreezzati saranno il 9, il 10 e l'11
        num_total_blocks = len(self.backbone.blocks)

        if (num_blocks == 12): 
            self.backbone.pos_embed.requires_grad = True
            self.backbone.cls_token.requires_grad = True
            for p in self.backbone.patch_embed.parameters():  # embedding vectors sono le rappresentazioni numeriche dei dati
                p.requires_grad = True
        else:
            for p in self.backbone.patch_embed.parameters():  # embedding vectors sono le rappresentazioni numeriche dei dati
                p.requires_grad = False
            self.backbone.pos_embed.requires_grad = False
            self.backbone.cls_token.requires_grad = False

        if num_blocks > num_total_blocks:
            print(f"Warning: Requested to unfreeze {num_blocks} blocks, but backbone only has {num_total_blocks}.")
            exit(0)
        elif num_blocks < 0:
            print(f"Warning: Requested to unfreeze {num_blocks} blocks.")
            exit(0)
        
        self.num_layers_frozen = num_total_blocks - num_blocks
        start_index = num_total_blocks - num_blocks # es. se voglio defreezare 5 layer -> 12-5 = 7 -> 7-8-9-10-11 unfreezzati
        for i in range(start_index, num_total_blocks): 
            block = self.backbone.blocks[i]
            block.train() 
            for param in block.parameters():
                param.requires_grad = True

        for i in range(num_total_blocks - num_blocks):
            block = self.backbone.blocks[i]
            block.eval()
            for param in block.parameters():
                param.requires_grad = False
