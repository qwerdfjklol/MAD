from functools import partial
import random
import typing as tp
import math
import torch
from torch import nn
from torch.nn import functional as F
# import torchaudio as ta
import numpy as np
import torch.nn.init as init
import pickle
from fairseq import utils
from .multihead_attention import MultiheadAttention
from torch import Tensor
from typing import Dict, List, Optional, Any
import contextlib
from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder
)
from fairseq.modules import (
    FairseqDropout,
    LayerNorm,
    TransformerEncoderLayer,
    PositionalEmbedding,
    LayerDropModuleList
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper


with open('/mnt/petrelfs/zhangchi1/m2t/layout.pkl', 'rb') as f:
    loaded_layout = pickle.load(f)


class PositionGetter:
    INVALID = -0.1

    def __init__(self) -> None:
        self._cache: tp.Dict[int, torch.Tensor] = {}
        self._invalid_names: tp.Set[str] = set()

    def get_recording_layout(self, layout) -> torch.Tensor:
        indexes: tp.List[int] = []
        valid_indexes: tp.List[int] = []
        for meg_index, name in enumerate(layout.names):
            name = name.rsplit("-", 1)[0]
            try:
                indexes.append(layout.names.index(name))
            except ValueError:
                if name not in self._invalid_names:
                    print(
                        "Channels %s not in layout for recording",
                        name,)
                    self._invalid_names.add(name)
            else:
                valid_indexes.append(meg_index)

        positions = torch.full((len(layout.names), 2), self.INVALID)
        x, y = layout.pos[indexes, :2].T
        x = (x - x.min()) / (x.max() - x.min())
        y = (y - y.min()) / (y.max() - y.min())
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        positions[valid_indexes, 0] = x
        positions[valid_indexes, 1] = y
        return positions

    def get_positions(self, meg, batch, layout):
        B, C, T = meg.shape
        positions = torch.full((B, C, 2), self.INVALID, device=meg.device)
        for idx in range(len(batch['subject_index'])):
            rec_pos = self.get_recording_layout(layout)
            positions[idx, :len(rec_pos)] = rec_pos.to(meg.device)
        return positions

    def is_invalid(self, positions):
        return (positions == self.INVALID).all(dim=-1)

class FourierEmb(nn.Module):
    """
    Fourier positional embedding.
    Unlike trad. embedding this is not using exponential periods
    for cosines and sinuses, but typical `2 pi k` which can represent
    any function over [0, 1]. As this function would be necessarily periodic,
    we take a bit of margin and do over [-0.2, 1.2].
    """

    def __init__(self, dimension: int = 256, margin: float = 0.2):
        super().__init__()
        n_freqs = (dimension // 2) ** 0.5
        assert int(n_freqs ** 2 * 2) == dimension
        self.dimension = dimension
        self.margin = margin

    def forward(self, positions):
        *O, D = positions.shape
        assert D == 2
        *O, D = positions.shape
        n_freqs = (self.dimension // 2) ** 0.5
        freqs_y = torch.arange(n_freqs).to(positions)
        freqs_x = freqs_y[:, None]
        width = 1 + 2 * self.margin
        positions = positions + self.margin
        p_x = 2 * math.pi * freqs_x / width
        p_y = 2 * math.pi * freqs_y / width
        positions = positions[..., None, None, :]
        loc = (positions[..., 0] * p_x + positions[..., 1] * p_y).view(*O, -1)
        emb = torch.cat([
            torch.cos(loc),
            torch.sin(loc),
        ], dim=-1)
        return emb

class ChannelMerger(nn.Module):
    def __init__(self, chout: int, pos_dim: int = 256,
                 dropout: float = 0, usage_penalty: float = 0.,
                 n_subjects: int = 200, per_subject: bool = False):
        super().__init__()
        assert pos_dim % 4 == 0
        self.position_getter = PositionGetter()
        self.per_subject = per_subject
        if self.per_subject:
            self.heads = nn.Parameter(torch.randn(n_subjects, chout, pos_dim, requires_grad=True))
        else:
            self.heads = nn.Parameter(torch.randn(chout, pos_dim, requires_grad=True))
        self.heads.data /= pos_dim ** 0.5
        self.dropout = dropout
        self.embedding = FourierEmb(pos_dim)
        self.usage_penalty = usage_penalty
        self._penalty = torch.tensor(0.)

    @property
    def training_penalty(self):
        return self._penalty.to(next(self.parameters()).device)

    def forward(self, meg, batch, layout):
        B, C, T = meg.shape
        meg = meg.clone()
        positions = self.position_getter.get_positions(meg, batch, layout)
        embedding = self.embedding(positions)
        score_offset = torch.zeros(B, C, device=meg.device)
        # score_offset[self.position_getter.is_invalid(positions)] = float('-inf')

        if self.training and self.dropout:
            center_to_ban = torch.rand(2, device=meg.device)
            radius_to_ban = self.dropout
            banned = (positions - center_to_ban).norm(dim=-1) <= radius_to_ban
            score_offset[banned] = float('-inf')
        if self.per_subject:
            _, cout, pos_dim = self.heads.shape
            subject = batch['subject_index']-1 # -1?
            heads = self.heads.gather(0, subject.view(-1, 1, 1).expand(-1, cout, pos_dim))
        else:
            heads = self.heads[None].expand(B, -1, -1)

        scores = torch.einsum("bcd,bod->boc", embedding, heads)
        scores += score_offset[:, None]
        weights = torch.softmax(scores, dim=2)
        out = torch.einsum("bct,boc->bot", meg, weights)
        if self.training and self.usage_penalty > 0.:
            usage = weights.mean(dim=(0, 1)).sum()
            self._penalty = self.usage_penalty * usage
        return out

class SubjectLayers(nn.Module):
    """Per subject linear layer."""

    def __init__(self, in_channels: int, out_channels: int, n_subjects: int, init_id: bool = False):
        super().__init__()
        self.C = in_channels
        self.D = out_channels
        self.weights = nn.Parameter(torch.randn(n_subjects, in_channels, out_channels), requires_grad=True)
        #self.weights = nn.Embedding(n_subjects, in_channels*out_channels)
        if init_id:
            assert in_channels == out_channels
            self.weights.data[:] = torch.eye(in_channels)[None]
        # self.weights.data *= 1 / in_channels ** 0.5
    def forward(self, x, subjects):
        B = x.shape[0]
        weights = self.weights.gather(0, subjects.long().view(-1, 1, 1).expand(-1, self.C, self.D))
        #weights = self.weights(subjects).view(B, self.C, self.D)
        return torch.einsum("bct,bcd->bdt", x, weights)

    def __repr__(self):
        S, C, D = self.weights.shape
        return f"SubjectLayers({C}, {D}, {S})"

class ConvPreNet(nn.Module):
    def __init__(self,
                 # Channels
                 in_channels: tp.Dict[str, int] = {"meg": 208},
                 out_channels: int = 80,
                 hidden: tp.Dict[str, int] = {"meg": 320},
                 # Overall structure
                 # Conv layer
                 gelu: bool = True,
                 relu_leakiness: float = 0.0,
                 # Subject specific settings
                 n_subjects: int = 27,
                 subject_layers: bool = True,
                 subject_layers_dim: str = "hidden",  # or hidden
                 subject_layers_id: bool = False,
                 # Attention multi-dataset support
                 merger: bool = True,
                 merger_pos_dim: int = 2048,
                 merger_channels: int = 270,
                 merger_dropout: float = 0.2,
                 merger_penalty: float = 0.,
                 merger_per_subject: bool = False,
                 dropout: float = 0.,
                 initial_linear: int = 270,
                 initial_depth: int = 1,
                 initial_nonlin: bool = False,
                 run_name = 'default'
                 ):
        super().__init__()
        self.run_name= run_name
        if set(in_channels.keys()) != set(hidden.keys()):
            raise ValueError("Channels and hidden keys must match "
                             f"({set(in_channels.keys())} and {set(hidden.keys())})")
        self.out_channels = out_channels
        if gelu:
            activation = nn.GELU
        elif relu_leakiness:
            activation = partial(nn.LeakyReLU, relu_leakiness)
        else:
            activation = nn.ReLU

        
        self.layout=loaded_layout # only gwilliams
        self.merger = None
        if merger:
            self.merger = ChannelMerger(
                merger_channels, pos_dim=merger_pos_dim, dropout=merger_dropout,
                usage_penalty=merger_penalty, n_subjects=n_subjects, per_subject=merger_per_subject)
            in_channels["meg"] = merger_channels

        self.initial_linear = None
        if initial_linear:
            init = [nn.Conv1d(in_channels["meg"], initial_linear, 1)]
            for _ in range(initial_depth - 1):
                init += [activation(), nn.Conv1d(initial_linear, initial_linear, 1)]
            if initial_nonlin:
                init += [activation()]
            self.initial_linear = nn.Sequential(*init)
            in_channels["meg"] = initial_linear

        self.subject_layers = None
        if subject_layers:
            assert "meg" in in_channels
            meg_dim = in_channels["meg"]
            dim = {"hidden": hidden["meg"], "input": meg_dim}[subject_layers_dim]
            self.subject_layers = SubjectLayers(meg_dim, dim, n_subjects, subject_layers_id)
            in_channels["meg"] = dim


        self.embed_positions = PositionalEmbedding(
                max_speech_positions, encoder_embed_dim
        )

    def forward(self, inputs, batch):
        # subj idx,
        subjects = batch['subject_index']-1
        length = inputs["meg"].shape[2]  # length of any of the inputs

        
        if self.merger is not None:
            inputs["meg"] = self.merger(inputs["meg"], batch, self.layout)
        
        if self.initial_linear is not None:
            inputs["meg"] = self.initial_linear(inputs["meg"])

        if self.subject_layers is not None:
            inputs["meg"] = self.subject_layers(inputs["meg"], subjects)


        if self.subject_embedding is not None:
            emb = self.subject_embedding(subjects)[:, :, None]
            inputs["meg"] = torch.cat([inputs["meg"], emb.expand(-1, -1, length)], dim=1)

        positions = self.embed_positions(encoder_padding_mask)
        x = inputs["meg"] + positions

        return x

class TransformerEncoder(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, tgt_dict=None, embed_tokens=None):
        self.args = args
        super().__init__(None)
        self.register_buffer("version", torch.Tensor([3]))

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.encoder_layerdrop = args.encoder_layerdrop
        self.freeze_encoder_updates = args.freeze_encoder_updates
        self.no_freeze_encoder_layer = None
        self.num_updates = 0
        export = getattr(args, "export", False)

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [self.build_encoder_layer(args) for i in range(args.encoder_layers)]
        )
        self.num_layers = len(self.layers)

        self.use_sent_enc_layer = args.use_sent_enc_layer
        self.unb_enc_layer = getattr(args, "unb_enc_layer", -1)

        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = LayerNorm(args.encoder_embed_dim, eps=args.layer_norm_eps, export=export)
        
        if args.share_ctc_embed and embed_tokens is not None:
            self.proj = nn.Linear(
                embed_tokens.weight.shape[1],
                embed_tokens.weight.shape[0],
                bias=False,
            )
            self.proj.weight = embed_tokens.weight
        elif tgt_dict is not None:
            self.proj = Linear(args.encoder_embed_dim, len(tgt_dict))
        else:
            self.proj = None
        
        if args.relative_position_embedding:
            self.pos_emb = RelativePositionalEncoding(args.encoder_embed_dim//args.encoder_attention_heads, args.encoder_max_relative_position)


    def build_encoder_layer(self, args):
        if args.use_sent_enc_layer:
            layer = TransformerSentenceEncoderLayer(
                embedding_dim=args.encoder_embed_dim,
                ffn_embedding_dim=args.encoder_ffn_embed_dim,
                num_attention_heads=args.encoder_attention_heads,
                dropout=args.dropout,
                attention_dropout=args.attention_dropout,
                activation_dropout=args.activation_dropout,
                activation_fn=args.activation_fn,
                layer_norm_first=args.layer_norm_first,
                has_relative_attention_bias=args.relative_position_embedding,
            )
        else:
            layer = TransformerEncoderLayer(args)
        return layer

    def forward(
        self,
        encoder_in,
        encoder_padding_mask,
        return_all_hiddens: bool = False,
        tgt_layer=None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        if self.no_freeze_encoder_layer is None:
            ft = self.freeze_encoder_updates <= self.num_updates
        else:
            ft = True
        with torch.no_grad() if not ft else contextlib.ExitStack():
            encoder_out = self.forward_scriptable(
                encoder_in, encoder_padding_mask, return_all_hiddens, tgt_layer=tgt_layer,
            )

        # CTC and bert
        if self.proj:
            x_for_ctc = self.proj(self.dropout_module(encoder_out["encoder_out"][0]))
        else:
            x_for_ctc = None

        encoder_out["encoder_out_for_ctc"] = [x_for_ctc] # T x B x C

        return encoder_out

    # TorchScript doesn't support super() method so that the scriptable Subclass
    # can't access the base class model in Torchscript.
    # Current workaround is to add a helper function with different name and
    # call the helper function from scriptable Subclass.
    def forward_scriptable(
        self,
        encoder_in,
        encoder_padding_mask,
        return_all_hiddens: bool = False,
        tgt_layer=None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        if self.no_freeze_encoder_layer is not None:
            ft = self.freeze_encoder_updates <= self.num_updates
        else:
            ft = True
        with torch.no_grad() if not ft else contextlib.ExitStack():
            # compute padding mask
            if not self.use_sent_enc_layer:
                has_pads = encoder_in.device.type == "xla" or encoder_padding_mask.any()

            if not self.layer_norm_first:
                encoder_in = self.layer_norm(encoder_in)

            encoder_in = self.dropout_module(encoder_in)

            # B x T x C -> T x B x C
            x = encoder_in.transpose(0, 1)

            encoder_states = []

            if return_all_hiddens:
                encoder_states.append(x)

            ## relative position embedding
            if self.args.relative_position_embedding:
                x_len = x.shape[0]
                pos_seq = torch.arange(0, x_len).long().to(x.device)
                pos_seq = pos_seq[:, None] - pos_seq[None, :]
                pos_k, pos_v = self.pos_emb(pos_seq)
            else:
                pos_k = None

        # encoder layers
        r = None
        d = None
        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random()

            with torch.no_grad() if (not ft) and i not in self.no_freeze_encoder_layer else contextlib.ExitStack():
                if not self.training or (dropout_probability > self.encoder_layerdrop) or i == self.unb_enc_layer:
                    if self.use_sent_enc_layer:
                        x, _ = layer(x, self_attn_padding_mask=encoder_padding_mask, self_attn_mask=None, need_weights=False, pos_bias=pos_k)
                    else:
                        x = layer(x, encoder_padding_mask=encoder_padding_mask if has_pads else None, attn_mask=None)
                if i == self.unb_enc_layer:
                    d = x

                if i == tgt_layer:
                    r = x
                    break

                if return_all_hiddens:
                    assert encoder_states is not None
                    encoder_states.append(x)

        with torch.no_grad() if not ft else contextlib.ExitStack():
            # Finally T x B x C
            if self.layer_norm_first:
                x = self.layer_norm(x.transpose(0, 1)).transpose(0, 1)

            if r is not None:
                x = r

        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "decoder_input": [d],
        }

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        for i in range(self.num_layers):
            # update layer norms
            if not isinstance(self.layers[i], TransformerSentenceEncoderLayer):
                self.layers[i].upgrade_state_dict_named(
                    state_dict, "{}.layers.{}".format(name, i)
                )

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates


class TransformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embedding_dim: float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: float = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        layer_norm_first: bool = False,
        has_relative_attention_bias: bool = False,
    ) -> None:

        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Initialize blocks
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
            has_relative_attention_bias=has_relative_attention_bias,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.layer_norm_first = layer_norm_first

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim)

        if has_relative_attention_bias:
            self.norm_k = LayerNorm(self.embedding_dim//num_attention_heads)

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
        att_args=None,
        pos_bias=None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            if pos_bias is not None:
                pos_bias = self.norm_k(pos_bias)
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                attn_mask=self_attn_mask,
                position_bias=pos_bias,
            )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
        else:
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                position_bias=pos_bias,
            )

            x = self.dropout1(x)
            x = residual + x

            x = self.self_attn_layer_norm(x)

            residual = x
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)

        return x, attn

class RelativePositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, maxlen=1000, embed_v=False):
        super(RelativePositionalEncoding, self).__init__()

        self.d_model = d_model
        self.maxlen = maxlen
        self.pe_k = torch.nn.Embedding(2*maxlen, d_model) 
        if embed_v:
            self.pe_v = torch.nn.Embedding(2*maxlen, d_model)
        self.embed_v = embed_v


    def forward(self, pos_seq):
        pos_seq[pos_seq < -self.maxlen] = -self.maxlen
        pos_seq[pos_seq >= self.maxlen] = self.maxlen - 1
        pos_seq = pos_seq + self.maxlen
        if self.embed_v:
            return self.pe_k(pos_seq), self.pe_v(pos_seq)
        else:
            return self.pe_k(pos_seq), None

class TransformerDecoderLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False, has_relative_attention_bias=False
    ):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.num_updates = 0
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        self.freeze_decoder_updates = getattr(args, "freeze_decoder_updates", 0)

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            args,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

        self.activation_fn = utils.get_activation_fn(
            activation=str(args.activation_fn)
            if getattr(args, "activation_fn", None) is not None
            else "relu"
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.decoder_normalize_before

        export = getattr(args, "export", False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.decoder_ffn_embed_dim,
        )
        self.fc2 = self.build_fc2(
            args.decoder_ffn_embed_dim,
            self.embed_dim,
        )

        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.need_attn = True

        self.has_relative_attention_bias = has_relative_attention_bias
        if self.has_relative_attention_bias:
            self.norm_k = LayerNorm(self.embed_dim//args.decoder_attention_heads)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return nn.Linear(input_dim, output_dim)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return nn.Linear(input_dim, output_dim)

    def build_self_attention(
        self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not getattr(args, "cross_self_attention", False),
        )

    def build_encoder_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
        )


    def residual_connection(self, x, residual):
        return residual + x

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
        pos_bias=None,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        ft = self.freeze_decoder_updates <= self.num_updates
    
        with torch.no_grad() if not ft else contextlib.ExitStack():
            if need_head_weights:
                need_attn = True

            residual = x
            if self.normalize_before:
                x = self.self_attn_layer_norm(x)
                if pos_bias is not None:
                    pos_bias = self.norm_k(pos_bias)
            if prev_self_attn_state is not None:
                prev_key, prev_value = prev_self_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_self_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
                assert incremental_state is not None
                self.self_attn._set_input_buffer(incremental_state, saved_state)
            _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
            if self.cross_self_attention and not (
                incremental_state is not None
                and _self_attn_input_buffer is not None
                and "prev_key" in _self_attn_input_buffer
            ):
                if self_attn_mask is not None:
                    assert encoder_out is not None
                    self_attn_mask = torch.cat(
                        (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                    )
                if self_attn_padding_mask is not None:
                    if encoder_padding_mask is None:
                        assert encoder_out is not None
                        encoder_padding_mask = self_attn_padding_mask.new_zeros(
                            encoder_out.size(1), encoder_out.size(0)
                        )
                    self_attn_padding_mask = torch.cat(
                        (encoder_padding_mask, self_attn_padding_mask), dim=1
                    )
                assert encoder_out is not None
                y = torch.cat((encoder_out, x), dim=0)
            else:
                y = x

            x, attn = self.self_attn(
                query=x,
                key=y,
                value=y,
                key_padding_mask=self_attn_padding_mask,
                incremental_state=incremental_state,
                need_weights=False,
                attn_mask=self_attn_mask,
                position_bias=pos_bias,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        with torch.no_grad() if not ft else contextlib.ExitStack():
            residual = x
            if self.normalize_before:
                x = self.final_layer_norm(x)

            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            x = self.fc2(x)
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.final_layer_norm(x)
            if incremental_state is not None:
                saved_state = self.self_attn._get_input_buffer(incremental_state)
                assert saved_state is not None
                if self_attn_padding_mask is not None:
                    self_attn_state = [
                        saved_state["prev_key"],
                        saved_state["prev_value"],
                        saved_state["prev_key_padding_mask"],
                    ]
                else:
                    self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
                return x, attn, self_attn_state
        return x, attn, None

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        self.num_updates = num_updates

class TransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self,
        args,
        no_encoder_attn=False,
    ):
        self.args = args
        super().__init__(None)
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask = torch.empty(0)

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.decoder_layerdrop = args.decoder_layerdrop
        export = getattr(args, "export", False)
        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        if self.decoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.decoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_decoder_layer(args, no_encoder_attn)
                for _ in range(args.decoder_layers)
            ]
        )
        self.num_layers = len(self.layers)

        if args.decoder_normalize_before and not getattr(
            args, "no_decoder_final_norm", False
        ):
            self.layer_norm = LayerNorm(args.decoder_embed_dim, eps=args.layer_norm_eps, export=export)
        else:
            self.layer_norm = None

        if args.relative_position_embedding:
            self.pos_emb = RelativePositionalEncoding(args.encoder_embed_dim//args.encoder_attention_heads, args.decoder_max_relative_position)

    def build_decoder_layer(self, args, no_encoder_attn=False):
        layer = TransformerDecoderLayer(args, no_encoder_attn=no_encoder_attn, has_relative_attention_bias=args.relative_position_embedding)
        return layer

    def forward(
        self,
        prev_output_tokens,
        tgt_mask,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention, should be of size T x B x C
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """

        x, extra = self.extract_features(
            prev_output_tokens,
            tgt_mask,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )

        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        tgt_mask,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            tgt_mask,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )

    """
    A scriptable subclass of this class has an extract_features method and calls
    super().extract_features, but super() is not supported in torchscript. A copy of
    this function is made to be used in the subclass instead.
    """

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        tgt_mask,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        bs = prev_output_tokens.size(0)
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
            assert (
                enc.size()[1] == bs
            ), f"Expected enc.shape == (t, {bs}, c) got {enc.shape}"
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]

        # B x T x C -> T x B x C
        x = prev_output_tokens.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or tgt_mask is not None:
            self_attn_padding_mask = tgt_mask

        ## relative position embedding
        if self.args.relative_position_embedding:
            x_len = x.shape[0]
            pos_seq = torch.arange(0, x_len).long().to(x.device)
            pos_seq = pos_seq[:, None] - pos_seq[None, :]
            pos_k, pos_v = self.pos_emb(pos_seq)
        else:
            pos_k = None

        # decoder layers
        attn_list = []
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer or alignment_layer == -1)),
                need_head_weights=bool((idx == alignment_layer or alignment_layer == -1)),
                pos_bias=pos_k,
            )
            inner_states.append(x)
            if layer_attn is not None and (idx == alignment_layer or alignment_layer == -1):
                attn = layer_attn.float().to(x)
                attn_list.append(attn.transpose(0, 1))

        if attn is not None and len(attn_list) == 1:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x, {"attn": [attn if len(attn_list) <= 1 else attn_list], "inner_states": inner_states}

    # def max_positions(self):
    #     """Maximum output length supported by the decoder."""
    #     return self.max_target_positions

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim, dim], device=tensor.device)), 1,
            )
        else:
            self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        for i in range(self.num_layers):
            # update layer norms
            layer_norm_map = {
                "0": "self_attn_layer_norm",
                "1": "encoder_attn_layer_norm",
                "2": "final_layer_norm",
            }
            for old, new in layer_norm_map.items():
                for m in ("weight", "bias"):
                    k = "{}.layers.{}.layer_norms.{}.{}".format(name, i, old, m)
                    if k in state_dict:
                        state_dict[
                            "{}.layers.{}.{}.{}".format(name, i, new, m)
                        ] = state_dict[k]
                        del state_dict[k]

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        return state_dict

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""

        def _apply(m):
            if hasattr(m, "set_num_updates") and m != self:
                m.set_num_updates(num_updates)

        self.apply(_apply)