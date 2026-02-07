"""
Model definitions for tabular data.
All models follow the same interface for easy switching.
"""
import torch
import torch.nn.functional as F
from types import SimpleNamespace
from typing import Any, Dict, List

from rllm.types import ColType
from rllm.nn.conv.table_conv import (
    FTTransformerConv,
    TabTransformerConv,
    ExcelFormerConv,
    SAINTConv,
    TromptConv,
)


class FTTransformer(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        layers: int,
        metadata: Dict[ColType, List[Dict[str, Any]]] = None,
        task: str = "classification",
    ):
        super().__init__()
        self.task = task
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            FTTransformerConv(
                conv_dim=hidden_dim,
                use_pre_encoder=True,
                metadata=metadata,
            )
        )
        for _ in range(layers - 1):
            self.convs.append(
                FTTransformerConv(conv_dim=hidden_dim)
            )

        self.fc = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, table_data: SimpleNamespace):
        x = table_data.feat_dict if hasattr(table_data, "feat_dict") else table_data
        for conv in self.convs:
            x = conv(x)
        out = self.fc(x[:, 0, :])
        if self.task == "regression":
            out = out.squeeze(-1)
        return out


class TabTransformer(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        layers: int,
        num_heads: int,
        metadata: Dict[ColType, List[Dict[str, Any]]] = None,
        task: str = "classification",
    ):
        super().__init__()
        self.task = task
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            TabTransformerConv(
                conv_dim=hidden_dim,
                num_heads=num_heads,
                use_pre_encoder=True,
                metadata=metadata,
            )
        )
        for _ in range(layers - 1):
            self.convs.append(
                TabTransformerConv(conv_dim=hidden_dim, num_heads=num_heads)
            )

        if metadata:
            num_cat = len(metadata.get(ColType.CATEGORICAL, []))
            num_num = len(metadata.get(ColType.NUMERICAL, []))
            fc_input_dim = num_cat * hidden_dim + num_num
        else:
            fc_input_dim = hidden_dim
        self.fc = torch.nn.Linear(fc_input_dim, out_dim)

    def forward(self, table_data: SimpleNamespace):
        x = table_data.feat_dict if hasattr(table_data, "feat_dict") else table_data
        for conv in self.convs:
            x = conv(x)
        if ColType.CATEGORICAL in x:
            x[ColType.CATEGORICAL] = x[ColType.CATEGORICAL].flatten(1)
        if ColType.NUMERICAL in x:
            x[ColType.NUMERICAL] = x[ColType.NUMERICAL].flatten(1)

        x = torch.cat(list(x.values()), dim=1)
        out = self.fc(x)
        if self.task == "regression":
            out = out.squeeze(-1)
        return out


class ExcelFormer(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        layers: int,
        metadata: Dict[ColType, List[Dict[str, Any]]] = None,
        task: str = "classification",
    ):
        super().__init__()
        self.task = task
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            ExcelFormerConv(
                conv_dim=hidden_dim,
                use_pre_encoder=True,
                metadata=metadata,
            )
        )
        for _ in range(layers - 1):
            self.convs.append(
                ExcelFormerConv(conv_dim=hidden_dim)
            )

        self.fc = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, table_data: SimpleNamespace):
        x = table_data.feat_dict if hasattr(table_data, "feat_dict") else table_data
        for conv in self.convs:
            x = conv(x)
        out = self.fc(x.mean(dim=1))
        if self.task == "regression":
            out = out.squeeze(-1)
        return out


class SAINT(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        num_feats: int,
        layers: int,
        metadata: Dict[ColType, List[Dict[str, Any]]] = None,
        task: str = "classification",
    ):
        super().__init__()
        self.task = task
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            SAINTConv(
                conv_dim=hidden_dim,
                num_feats=num_feats,
                use_pre_encoder=True,
                metadata=metadata,
            )
        )
        for _ in range(layers - 1):
            self.convs.append(
                SAINTConv(
                    conv_dim=hidden_dim,
                    num_feats=num_feats,
                )
            )
        self.fc = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, table_data: SimpleNamespace):
        x = table_data.feat_dict if hasattr(table_data, "feat_dict") else table_data
        for conv in self.convs:
            x = conv(x)
        out = self.fc(x.mean(dim=1))
        if self.task == "regression":
            out = out.squeeze(-1)
        return out


class TromptNet(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        layers: int,
        num_prompts: int = 128,
        metadata: Dict[ColType, List[Dict[str, Any]]] = None,
        task: str = "classification",
    ):
        super().__init__()
        self.task = task
        self.out_dim = out_dim
        self.x_prompt = torch.nn.Parameter(torch.empty(num_prompts, hidden_dim))

        self.convs = torch.nn.ModuleList()
        for _ in range(layers):
            self.convs.append(
                TromptConv(
                    in_dim=in_dim,
                    out_dim=hidden_dim,
                    num_prompts=num_prompts,
                    use_pre_encoder=True,
                    metadata=metadata,
                )
            )
        self.linear = torch.nn.Linear(hidden_dim, 1)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, out_dim),
        )
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.x_prompt)
        for conv in self.convs:
            conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)
        for layer in self.mlp:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

    def forward(self, table_data: SimpleNamespace):
        x = table_data.feat_dict if hasattr(table_data, "feat_dict") else table_data
        batch_size = x[list(x.keys())[0]].size(0)
        x_prompt = self.x_prompt.unsqueeze(0).repeat(batch_size, 1, 1)
        outs = []
        for conv in self.convs:
            x_prompt = conv(x, x_prompt)
            w_prompt = F.softmax(self.linear(x_prompt), dim=1)
            out = (w_prompt * x_prompt).sum(dim=1)
            out = self.mlp(out)
            out = out.reshape(batch_size, 1, self.out_dim)
            outs.append(out)

        out = torch.cat(outs, dim=1).mean(dim=1)
        if self.task == "regression":
            out = out.squeeze(-1)
        return out


# ======================== Model Factory ========================
def create_model(
    model_name: str,
    config: Dict,
    data,
    device: torch.device,
    task: str = "classification",
):
    """
    Factory function to create models.
    
    Args:
        model_name: Name of the model ("fttransformer", "tabtransformer", etc.)
        config: Dictionary with model hyperparameters
        data: Dataset object with metadata
        device: Device to place model on
        task: "classification" or "regression"
    
    Returns:
        Model instance
    """
    model_name = model_name.lower()
    
    # Determine output dimension
    if task == "classification":
        out_dim = data.num_classes
    else:
        out_dim = 1
    
    # Get common parameters
    hidden_dim = config.get("hidden_dim", 32)
    layers = config.get("layers", 3)
    metadata = data.metadata
    
    # Create model based on name
    if model_name == "fttransformer":
        model = FTTransformer(
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            layers=layers,
            metadata=metadata,
            task=task,
        )
    
    elif model_name == "tabtransformer":
        model = TabTransformer(
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            layers=layers,
            num_heads=config.get("num_heads", 8),
            metadata=metadata,
            task=task,
        )
    
    elif model_name == "excelformer":
        model = ExcelFormer(
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            layers=layers,
            metadata=metadata,
            task=task,
        )
    
    elif model_name == "saint":
        # Compute num_feats from metadata
        if metadata is not None:
            num_feats = sum(len(v) for v in metadata.values())
        else:
            num_feats = hidden_dim
        
        model = SAINT(
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_feats=num_feats,
            layers=layers,
            metadata=metadata,
            task=task,
        )
    
    elif model_name == "tromptnet":
        # Compute in_dim from metadata
        if metadata is not None:
            in_dim = sum(len(cols) for cols in metadata.values())
        else:
            in_dim = hidden_dim
        
        model = TromptNet(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            layers=layers,
            num_prompts=config.get("num_prompts", 128),
            metadata=metadata,
            task=task,
        )
    
    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: fttransformer, tabtransformer, excelformer, saint, tromptnet"
        )
    
    return model.to(device)


# Model registry for easy access
AVAILABLE_MODELS = [
    "fttransformer",
    "tabtransformer", 
    "excelformer",
    "saint",
    "tromptnet",
]

