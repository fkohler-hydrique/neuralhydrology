import itertools
import logging
from typing import Dict, Optional, Union, Tuple

import torch
import torch.nn as nn

from neuralhydrology.modelzoo.fc import FC
from neuralhydrology.modelzoo.positional_encoding import PositionalEncoding
from neuralhydrology.utils.config import Config

LOGGER = logging.getLogger(__name__)

_EMBEDDING_TYPES = ["full_model", "hindcast", "forecast"]


class InputLayer(nn.Module):
    """Input layer to preprocess static and dynamic inputs.

    Responsibilities
    ----------------
    - Select correct dynamic input groups depending on `embedding_type`:
      * 'full_model' → cfg.dynamic_inputs
      * 'hindcast'   → cfg.hindcast_inputs
      * 'forecast'   → cfg.forecast_inputs
    - Optionally apply:
      * fully-connected embeddings for dynamics/statics,
      * NaN handling (masked_mean / attention / input_replacing),
      * timestep counters (hindcast_counter / forecast_counter),
      * UMAL '_tau' input.

    Note
    ----
    - Multi-frequency configs (dynamic_inputs as dict with >1 key) are *not* supported.
    """

    def __init__(self, cfg: Config, embedding_type: str = "full_model") -> None:
        super().__init__()

        self.cfg = cfg
        self.embedding_type = embedding_type

        # NaN handling setup (used in all branches)
        self.nan_handling_method = cfg.nan_handling_method
        self.attention: nn.MultiheadAttention | None = None
        self._nan_fill_value: float = 0.0

        # ------------------------------------------------------------------ #
        # Choose dynamic inputs based on embedding_type
        # ------------------------------------------------------------------ #
        if embedding_type == "full_model":
            dynamic_inputs = cfg.dynamic_inputs
            self._x_d_key = "x_d"
        elif embedding_type == "forecast":
            dynamic_inputs = cfg.forecast_inputs
            self._x_d_key = "x_d_forecast"
        elif embedding_type == "hindcast":
            dynamic_inputs = cfg.hindcast_inputs
            self._x_d_key = "x_d_hindcast"
        else:
            raise ValueError(
                f"Embedding type {embedding_type} is not recognized. "
                f"Must be one of: {_EMBEDDING_TYPES}."
            )

        # ------------------------------------------------------------------ #
        # Organize dynamic input feature groups and compute input sizes
        # ------------------------------------------------------------------ #
        if isinstance(dynamic_inputs, dict):
            # Multi-frequency case: only single frequency supported here.
            if self.nan_handling_method:
                raise ValueError(
                    "InputLayer does not support nan handling methods with multiple frequencies."
                )
            frequencies = list(dynamic_inputs.keys())
            if len(frequencies) > 1:
                raise ValueError("InputLayer only supports single-frequency data.")

            # Example: {'1H': ['feat1', 'feat2']} → one group
            dynamics_input_sizes = [len(dynamic_inputs[frequencies[0]])]
            # Store as dict with one entry: {'1H': [['feat1', 'feat2']]}
            self._dynamic_inputs = {k: [v] for k, v in dynamic_inputs.items()}
        else:
            # dynamic_inputs is list-of-features or list-of-groups
            if isinstance(dynamic_inputs[0], str):
                # Put all features into a single feature group.
                self._dynamic_inputs = [dynamic_inputs]
            else:
                self._dynamic_inputs = dynamic_inputs

            # Add timestep counters if requested
            if cfg.timestep_counter:
                if self.embedding_type == "hindcast":
                    self._dynamic_inputs = [
                        group + ["hindcast_counter"] for group in self._dynamic_inputs
                    ]
                elif self.embedding_type == "forecast":
                    self._dynamic_inputs = [
                        group + ["forecast_counter"] for group in self._dynamic_inputs
                    ]

            # Determine input sizes per dynamic feature group for NaN-handling modes
            if self.nan_handling_method == "input_replacing":
                # +1 for per-group NaN flag + optional pos-encoding size (added later)
                dynamics_input_sizes = [
                    sum(len(group) + 1 for group in self._dynamic_inputs)
                    + cfg.nan_handling_pos_encoding_size
                ]
            elif self.nan_handling_method in ["masked_mean", "attention"]:
                dynamics_input_sizes = [
                    len(group)
                    + (
                        cfg.nan_handling_pos_encoding_size
                        if self.nan_handling_method != "attention"
                        else 0
                    )
                    for group in self._dynamic_inputs
                ]
            else:
                dynamics_input_sizes = [len(group) for group in self._dynamic_inputs]

        # UMAL: add '_tau' to inputs
        if cfg.head.lower() == "umal":
            dynamics_input_sizes = [size + 1 for size in dynamics_input_sizes]
            if isinstance(self._dynamic_inputs, dict):
                self._dynamic_inputs = {k: v + ["_tau"] for k, v in self._dynamic_inputs.items()}
            else:
                self._dynamic_inputs = [group + ["_tau"] for group in self._dynamic_inputs]

        # Autoregressive inputs are appended at the end (unembedded)
        self._num_autoregression_inputs = len(cfg.autoregressive_inputs or [])

        # ------------------------------------------------------------------ #
        # Static inputs: attributes + HydroATLAS + evolving + optional basin one-hot
        # ------------------------------------------------------------------ #
        statics_input_size = len(cfg.static_attributes + cfg.hydroatlas_attributes + cfg.evolving_attributes)
        if cfg.use_basin_id_encoding:
            statics_input_size += cfg.number_of_basins

        self.statics_embedding, self.statics_output_size = self._get_embedding_net(
            cfg.statics_embedding, statics_input_size, "statics"
        )

        # ------------------------------------------------------------------ #
        # Positional encoding (for NaN handling modes only)
        # ------------------------------------------------------------------ #
        self._pos_enc: PositionalEncoding | None = None
        if cfg.nan_handling_pos_encoding_size > 0:
            if not self.nan_handling_method:
                raise NotImplementedError(
                    "Positional encoding is only supported for nan handling methods."
                )
            self._pos_enc = PositionalEncoding(
                embedding_dim=cfg.nan_handling_pos_encoding_size,
                position_type="concatenate",
                dropout=0.0,
                max_len=cfg.seq_length,
            )

        # ------------------------------------------------------------------ #
        # Dynamics embedding nets (one per feature group)
        # ------------------------------------------------------------------ #
        dynamics_embeddings: list[nn.Module] = []
        dynamics_output_sizes: list[int] = []

        for dynamics_input_size in dynamics_input_sizes:
            group_embedding, group_output_size = self._get_embedding_net(
                cfg.dynamics_embedding, dynamics_input_size, "dynamics"
            )
            dynamics_embeddings.append(group_embedding)
            dynamics_output_sizes.append(group_output_size)

        self.dynamics_embeddings = nn.ModuleList(dynamics_embeddings)
        if not all(size == dynamics_output_sizes[0] for size in dynamics_output_sizes):
            raise ValueError("All dynamics embedding output sizes must be equal.")
        self.dynamics_output_size = dynamics_output_sizes[0]

        # Attention-based NaN handling: init attention + query embedding
        if self.nan_handling_method == "attention":
            self.attention = nn.MultiheadAttention(embed_dim=self.dynamics_output_size, num_heads=1)
            self.query_embedding, _ = self._get_embedding_net(
                cfg.dynamics_embedding,
                self.statics_output_size + len(self._dynamic_inputs) + cfg.nan_handling_pos_encoding_size,
                "query",
            )

        # Store dropout rates of embedding nets (used for fine-tuning strategies)
        if cfg.statics_embedding is None:
            self.statics_embedding_p_dropout = 0.0
        else:
            self.statics_embedding_p_dropout = cfg.statics_embedding["dropout"]

        if cfg.dynamics_embedding is None:
            self.dynamics_embedding_p_dropout = 0.0
        else:
            self.dynamics_embedding_p_dropout = cfg.dynamics_embedding["dropout"]

        # Final output size seen by the LSTM / model
        self.output_size = (
            self.dynamics_output_size + self.statics_output_size + self._num_autoregression_inputs
        )

    # ------------------------------------------------------------------ #
    # Embedding-net builder
    # ------------------------------------------------------------------ #
    @staticmethod
    def _get_embedding_net(
        embedding_spec: Optional[dict], input_size: int, purpose: str
    ) -> Tuple[nn.Module, int]:
        """Get an embedding net following the passed specifications.

        If `embedding_spec` is None, returns identity and the unchanged input_size.
        """
        if embedding_spec is None:
            return nn.Identity(), input_size

        if input_size == 0:
            raise ValueError(f"Cannot create {purpose} embedding layer with input size 0.")

        emb_type = embedding_spec["type"].lower()
        if emb_type != "fc":
            raise ValueError(f"{purpose} embedding type {emb_type} not supported.")

        hiddens = embedding_spec["hiddens"]
        if len(hiddens) == 0:
            raise ValueError(
                f'{purpose} embedding "hiddens" must be a list of hidden sizes with at least one entry.'
            )

        dropout = embedding_spec["dropout"]
        activation = embedding_spec["activation"]

        emb_net = FC(input_size=input_size, hidden_sizes=hiddens, activation=activation, dropout=dropout)
        return emb_net, emb_net.output_size

    # ------------------------------------------------------------------ #
    # Forward pass
    # ------------------------------------------------------------------ #
    def forward(
        self,
        data: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        concatenate_output: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Perform a forward pass on the input layer.

        Parameters
        ----------
        data : dict
            Input data containing 'x_d*', 'x_s', 'x_one_hot', etc.
        concatenate_output : bool, optional
            If True (default), concatenate static embedding to each dynamic timestep.
            If False, return (dynamics_out, statics_out) separately.

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            - If concatenate_output is True: (seq_len, batch, features_total)
            - Else: ( (seq_len, batch, dyn_features), (batch, static_features) )
        """
        # Resolve feature group list
        features = self._dynamic_inputs
        if isinstance(features, dict):
            features = features[list(features.keys())[0]]

        # Static features: x_s and/or x_one_hot
        if "x_s" in data and "x_one_hot" in data:
            x_s = torch.cat([data["x_s"], data["x_one_hot"]], dim=-1)
        elif "x_s" in data:
            x_s = data["x_s"]
        elif "x_one_hot" in data:
            x_s = data["x_one_hot"]
        else:
            x_s = None

        statics_out = None
        if x_s is not None:
            statics_out = self.statics_embedding(x_s)  # (batch, statics_output_size)

        # Dynamic features with NaN-handling
        if self.nan_handling_method == "masked_mean":
            dynamics_out = self._masked_mean_embedding(data[self._x_d_key])
        elif self.nan_handling_method == "attention":
            dynamics_out = self._attention(data[self._x_d_key], statics_embedding=statics_out)
        elif self.nan_handling_method == "input_replacing":
            dynamics_out = self._input_replacing_embedding(data[self._x_d_key])
        else:
            # Default: concatenate all dynamic groups and embed
            # shape: (seq_len, batch, n_features)
            x_d = torch.cat(
                [data[self._x_d_key][k] for k in itertools.chain(*features)],
                dim=-1,
            ).transpose(0, 1)
            dynamics_out = self.dynamics_embeddings[0](x_d)

        # No concatenation: return (dynamics, statics)
        if not concatenate_output:
            return dynamics_out, statics_out

        # Concatenate statics to each timestep
        if statics_out is not None:
            # dynamics_out: (seq_len, batch, dyn_dim)
            # statics_out: (batch, static_dim) → broadcast over seq_len
            statics_broadcast = statics_out.unsqueeze(0).repeat(dynamics_out.shape[0], 1, 1)
            ret_val = torch.cat([dynamics_out, statics_broadcast], dim=-1)
        else:
            ret_val = dynamics_out

        # Append autoregressive inputs at the end (unembedded)
        if self._num_autoregression_inputs:
            x_autoregressive = torch.cat(
                [data[self._x_d_key][k] for k in self.cfg.autoregressive_inputs],
                dim=-1,
            ).transpose(0, 1)
            ret_val = torch.cat([ret_val, x_autoregressive], dim=-1)

        return ret_val

    # ------------------------------------------------------------------ #
    # NaN-handling strategies
    # ------------------------------------------------------------------ #
    def _attention(
        self,
        x_d: dict[str, torch.Tensor],
        statics_embedding: torch.Tensor | None,
    ) -> torch.Tensor:
        """Attention mechanism with statics + positional encoding as query, feature groups as keys/values."""
        if statics_embedding is None:
            raise ValueError("Attention NaN handling requires static features.")

        dynamics_out = []
        masks = []

        for idx, feature_group in enumerate(self._dynamic_inputs):
            # -> (seq_len, batch, n_features)
            x_d_group = torch.cat([x_d[k] for k in feature_group], dim=-1).transpose(0, 1)
            mask = x_d_group.isnan().any(dim=-1, keepdim=True)

            # Zero out NaNs (ignored via mask)
            x_d_group = torch.where(mask, 0.0, x_d_group)
            group_embedding = self.dynamics_embeddings[idx](x_d_group)
            dynamics_out.append(torch.where(mask, torch.nan, group_embedding))
            masks.append(mask)

        dynamics_out = torch.stack(dynamics_out, dim=0)  # (n_groups, seq_len, batch, embed_dim)
        dynamics_out = torch.where(
            torch.isnan(dynamics_out).all(dim=0, keepdim=True),
            self._nan_fill_value,
            dynamics_out,
        )

        n_groups, seq_len, batch_size, embed_dim = dynamics_out.shape
        stacked_masks = torch.stack(masks, dim=0).view(n_groups, seq_len * batch_size, 1)

        # Query: (seq_len, batch, statics+pos+flags)
        query = statics_embedding.unsqueeze(0).repeat(seq_len, 1, 1)
        if self._pos_enc is not None:
            query = self._pos_enc(query)

        query = torch.cat(
            [query, stacked_masks.squeeze(-1).permute(1, 0).view(seq_len, batch_size, n_groups)],
            dim=-1,
        )
        query = self.query_embedding(query)
        query = query.unsqueeze(0).view(1, seq_len * batch_size, embed_dim)

        # Key/value: (n_groups, seq_len * batch_size, embed_dim)
        stacked_masks = stacked_masks.squeeze(-1)
        key = dynamics_out.view(n_groups, seq_len * batch_size, embed_dim)
        value = dynamics_out.view(n_groups, seq_len * batch_size, embed_dim)
        key = torch.where(stacked_masks.unsqueeze(2), self._nan_fill_value, key)
        value = torch.where(stacked_masks.unsqueeze(2), self._nan_fill_value, value)

        # attn_mask: (seq_len * batch_size, 1, n_groups)
        attn_mask = stacked_masks.permute(1, 0).unsqueeze(1)
        attention_out, _ = self.attention(
            query, key, value, attn_mask=attn_mask, need_weights=False
        )  # (1, seq_len * batch_size, embed_dim)

        return attention_out.view(1, seq_len, batch_size, embed_dim).squeeze(0)

    def _masked_mean_embedding(self, x_d: dict[str, torch.Tensor]) -> torch.Tensor:
        """Masked-mean embedding across feature groups."""
        dynamics_out = []
        masks = []

        for idx, feature_group in enumerate(self._dynamic_inputs):
            x_d_group = torch.cat([x_d[k] for k in feature_group], dim=-1).transpose(0, 1)
            mask = x_d_group.isnan().any(dim=-1, keepdim=True)
            if self._pos_enc is not None:
                x_d_group = self._pos_enc(x_d_group)
            x_d_group = torch.where(mask, 0.0, x_d_group)
            group_embedding = self.dynamics_embeddings[idx](x_d_group)
            dynamics_out.append(torch.where(mask, torch.nan, group_embedding))
            masks.append(mask)

        dynamics_out = torch.stack(dynamics_out, dim=0)
        dynamics_out = torch.where(
            torch.isnan(dynamics_out).all(dim=0, keepdim=True),
            self._nan_fill_value,
            dynamics_out,
        )
        return torch.nanmean(dynamics_out, dim=0)

    def _input_replacing_embedding(self, x_d: dict[str, torch.Tensor]) -> torch.Tensor:
        """Adds input masks to the inputs and sets NaNs to a constant fill value."""
        dynamics = []
        for feature_group in self._dynamic_inputs:
            x_d_group = torch.cat([x_d[k] for k in feature_group], dim=-1).transpose(0, 1)
            mask = x_d_group.isnan().any(dim=-1, keepdim=True)
            x_d_group = torch.where(mask, self._nan_fill_value, x_d_group)
            dynamics.append(x_d_group)
            dynamics.append(mask.to(torch.float32))

        dynamics = torch.cat(dynamics, dim=-1)
        if self._pos_enc is not None:
            dynamics = self._pos_enc(dynamics)

        return self.dynamics_embeddings[0](dynamics)

    # ------------------------------------------------------------------ #
    # Dict-like access for fine-tuning code
    # ------------------------------------------------------------------ #
    def __getitem__(self, item: str) -> nn.Module:
        """Allow dict-like access to submodules (used in fine-tuning utilities)."""
        if item == "statics_embedding":
            return self.statics_embedding
        if item == "dynamics_embeddings":
            return self.dynamics_embeddings
        raise KeyError(f"Cannot access {item} on InputLayer")
