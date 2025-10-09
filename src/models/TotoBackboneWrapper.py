import json
import os
import re
from pathlib import Path
from typing import Dict, Optional, Union
from jaxtyping import Bool, Float

import safetensors.torch as safetorch
from huggingface_hub import ModelHubMixin, constants, hf_hub_download
import torch 
from einops import rearrange

from toto.model.attention import XFORMERS_AVAILABLE
from toto.model.backbone import TotoBackbone
from toto.model.transformer import XFORMERS_SWIGLU_AVAILABLE
from toto.model.backbone import TotoOutput
from toto.model.util import KVCache

from src.models.CallsignEmbedding import CallsignEmbedding
from src.models.MultiScaleTimeEmbedding import MultiScaleTimeEmbedding


class TotoBackboneWrapper(torch.nn.Module, ModelHubMixin):

    def __init__(
        self,
        patch_size: int,
        stride: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_hidden_dim: int,
        dropout: float,
        spacewise_every_n_layers: int,
        scaler_cls: str,
        output_distribution_classes: list[str],
        spacewise_first: bool = True,
        output_distribution_kwargs: dict | None = None,
        use_memory_efficient_attention: bool = True,
        stabilize_with_global: bool = True,
        scale_factor_exponent: float = 10.0,
        callsign_vocab_size=None,
        time_embedding_scales=None,
        time_embedding_ref=None
    ):
        super().__init__()

        # Add custom components with double underscore prefix.
        # These are not loaded from checkpoints,
        # so we need to filter them out in state_dict and loading logic.
        self.__callsign_embedding = CallsignEmbedding(
            vocab_size=callsign_vocab_size,
            d_model=embed_dim
        )

        self.__time_embedding = MultiScaleTimeEmbedding(
            d_model=embed_dim,
            ref_timestamp=time_embedding_ref,
            scales=time_embedding_scales
        )

        self.model = TotoBackbone(
            patch_size=patch_size,
            stride=stride,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_hidden_dim=mlp_hidden_dim,
            dropout=dropout,
            spacewise_every_n_layers=spacewise_every_n_layers,
            scaler_cls=scaler_cls,
            output_distribution_classes=output_distribution_classes,
            spacewise_first=spacewise_first,
            output_distribution_kwargs=output_distribution_kwargs,
            use_memory_efficient_attention=use_memory_efficient_attention,
            stabilize_with_global=stabilize_with_global,
            scale_factor_exponent=scale_factor_exponent
            )

    def forward(
        self,
        inputs: Float[torch.Tensor, "batch variate time_steps"],
        input_padding_mask: Bool[torch.Tensor, "batch variate time_steps"],
        id_mask: Float[torch.Tensor, "batch #variate time_steps"],
        kv_cache: Optional[KVCache] = None,
        scaling_prefix_length: Optional[int] = None,
        timestamp: Optional[torch.Tensor] = None,
        callsigns: Optional[torch.Tensor] = None 
    ) -> TotoOutput:
        """
        input_padding_mask: (False where value is NaN/imputed/pad) - used in the scaler and in the loss
        id_mask: 
        The ID mask is used for packing unrelated time series along the Variate dimension.
        This is used in training, and can also be useful for large-scale batch inference in order to
        process time series of different numbers of variates using batches of a fixed shape.
        The ID mask controls the channel-wise attention; variates with different IDs cannot attend to each other.
        If you're not using packing, just set this to zeros.
        """

        scaled_inputs: Float[torch.Tensor, "batch variate time_steps"]
        loc: Float[torch.Tensor, "batch variate time_steps"]
        scale: Float[torch.Tensor, "batch variate time_steps"]

        # Standard scaling operation, same API but without ID mask.
        scaled_inputs, loc, scale = self.model.scaler(
            inputs,
            weights=torch.ones_like(inputs, device=inputs.device),
            padding_mask=input_padding_mask,
            prefix_length=scaling_prefix_length,
        )

        if kv_cache is not None:

            prefix_len = self.model.patch_embed.stride * kv_cache.current_len(0)

            # Truncate inputs so that the transformer only processes
            # the last patch in the sequence. We'll use the KVCache
            # for the earlier patches.
            scaled_inputs = scaled_inputs[:, :, prefix_len:]

            # As a simplification, when using kv cache we only allow decoding
            # one step at a time after the initial forward pass.
            assert (prefix_len == 0) or (
                scaled_inputs.shape[-1] == self.model.patch_embed.stride
            ), "Must decode one step at a time."

            input_padding_mask = input_padding_mask[:, :, prefix_len:]
            id_mask = id_mask[:, :, prefix_len:]

        embeddings: Float[torch.Tensor, "batch variate seq_len embed_dim"]
        reduced_id_mask: Float[torch.Tensor, "batch variate seq_len"]

        embeddings, reduced_id_mask = self.model.patch_embed(scaled_inputs, id_mask)

        callsign_token = self.__callsign_embedding(callsigns)  # [batch, d_model]
        time_embedding_token = self.__time_embedding(timestamp)  # taking only first timestamp and converting to seconds

        # Fuse callsign and time embedding to create the init token
        init_token = callsign_token + time_embedding_token  # [batch, embed_dim]

        # Add init_token as a new token at the start of each sequence in embeddings
        # embeddings: [batch, variate, seq_len, embed_dim]
        B, V, S, D = embeddings.shape
        init_token_expanded = init_token.unsqueeze(1).unsqueeze(2).expand(B, V, 1, D)  # [batch, variate, 1, embed_dim]
        embeddings = torch.cat([init_token_expanded, embeddings], dim=2)  # [batch, variate, seq_len+1, embed_dim]

        # Update reduced_id_mask to account for the new token
        init_mask = torch.zeros(B, V, 1, device=reduced_id_mask.device, dtype=reduced_id_mask.dtype)
        reduced_id_mask = torch.cat([init_mask, reduced_id_mask], dim=2)  # [batch, variate, seq_len+1]

        # Apply the transformer on the embeddings
        transformed: Float[torch.Tensor, "batch variates seq_len embed_dim"] = self.model.transformer(
            embeddings, reduced_id_mask, kv_cache
        )

        transformed = transformed[:, :, 1:, :]  # removing the init token from results

        # Unembed and flatten the sequence
        flattened: Float[torch.Tensor, "batch variates new_seq_len embed_dim"] = rearrange(
            self.model.unembed(transformed),
            "batch variates seq_len (patch_size embed_dim) -> batch variates (seq_len patch_size) embed_dim",
            embed_dim=self.model.embed_dim,
        )

        return TotoOutput(self.model.output_distribution(flattened), loc, scale)

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path,
        map_location: str = "cpu",
        strict=True,
        **model_kwargs,
    ):
        """
        Custom checkpoint loading. Used to load a local
        safetensors checkpoint with an optional config.json file.
        """
        if os.path.isdir(checkpoint_path):
            safetensors_file = os.path.join(checkpoint_path, "model.safetensors")
        else:
            safetensors_file = checkpoint_path

        if os.path.exists(safetensors_file):
            model_state = safetorch.load_file(safetensors_file, device=map_location)
        else:
            raise FileNotFoundError(f"Model checkpoint not found at: {safetensors_file}")

        # Load configuration from config.json if it exists.
        config_file = os.path.join(checkpoint_path, "config.json")
        config = {}
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                config = json.load(f)

        # Merge any extra kwargs into the configuration.
        config.update(model_kwargs)

        remapped_state_dict = cls._map_state_dict_keys(
            model_state, XFORMERS_SWIGLU_AVAILABLE and not config.get("pre_xformers_checkpoint", False)
        )

        if not XFORMERS_AVAILABLE and config.get("use_memory_efficient_attention", True):
            config["use_memory_efficient_attention"] = False

        instance = cls(**config)
        instance.to(map_location)

        # Filter out unexpected keys
        filtered_remapped_state_dict = {
            k: v
            for k, v in remapped_state_dict.items()
            if k in instance.state_dict() and not k.endswith("rotary_emb.freqs")
        }
        # Ensure any new parameters defined directly within TotoBackboneWrapper are included with their default-initialized values to satisfy strict=True. 
        filtered_remapped_state_dict.update({
            k:v for k, v in instance.state_dict().items() if k.startswith('_TotoBackboneWrapper__')
        })

        instance.load_state_dict(filtered_remapped_state_dict, strict=strict)
        return instance

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: Optional[str],
        cache_dir: Optional[Union[str, Path]],
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: Optional[bool],
        local_files_only: bool,
        token: Union[str, bool, None],
        map_location: str = "cpu",
        strict: bool = False,
        **model_kwargs,
    ):
        """Load Pytorch pretrained weights and return the loaded model."""
        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            model_file = os.path.join(model_id, constants.SAFETENSORS_SINGLE_FILE)
            return cls.load_from_checkpoint(model_file, map_location, strict, **model_kwargs)
        else:
            model_file = hf_hub_download(
                repo_id=model_id,
                filename=constants.SAFETENSORS_SINGLE_FILE,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )
            return cls.load_from_checkpoint(model_file, map_location, strict, **model_kwargs)

    @staticmethod
    def _map_state_dict_keys(state_dict, use_fused_swiglu):
        """
        Maps the keys of a state_dict to match the current model's state_dict.
        Currently this is only used to convert between fused and unfused SwiGLU implementations.
        """
        if use_fused_swiglu:
            remap_keys = {
                "mlp.0.weight": "mlp.0.w12.weight",
                "mlp.0.bias": "mlp.0.w12.bias",
                "mlp.2.weight": "mlp.0.w3.weight",
                "mlp.2.bias": "mlp.0.w3.bias",
            }
        else:
            remap_keys = {
                "mlp.0.w12.weight": "mlp.0.weight",
                "mlp.0.w12.bias": "mlp.0.bias",
                "mlp.0.w3.weight": "mlp.2.weight",
                "mlp.0.w3.bias": "mlp.2.bias",
            }

        def replace_key(text):
            for pattern, replacement in remap_keys.items():
                text = re.sub(pattern, replacement, text)
            return text

        return {replace_key(k): v for k, v in state_dict.items()}

