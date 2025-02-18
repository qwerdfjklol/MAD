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
from utils.transformer.multihead_attention import MultiheadAttention
from torch import Tensor
from typing import Dict, List, Optional, Any
import contextlib
from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder
)
from fairseq.modules.conformer_layer import ConformerEncoderLayer
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

class ConvSequence(nn.Module):

    def __init__(self, channels: tp.Sequence[int], kernel: int = 3, dilation_growth: int = 2,
                 dilation_period: tp.Optional[int] = 5, stride: int = 1,
                 dropout: float = 0.0, leakiness: float = 0.0, groups: int = 1,
                 decode: bool = False, batch_norm: bool = True, dropout_input: float = 0.0,
                 skip: bool = True, scale: tp.Optional[float] = None, rewrite: bool = False,
                 activation_on_last: bool = True, post_skip: bool = False, glu: int = 2,
                 glu_context: int = 1, glu_glu: bool = True, activation: tp.Any = nn.GELU) -> None:
        super().__init__()
        dilation = 1
        channels = tuple(channels)
        self.skip = skip
        self.sequence = nn.ModuleList()
        self.glus = nn.ModuleList()
        if activation is None:
            activation = partial(nn.LeakyReLU, leakiness)
        Conv = nn.Conv1d if not decode else nn.ConvTranspose1d
        # build layers
        for k, (chin, chout) in enumerate(zip(channels[:-1], channels[1:])):
            layers: tp.List[nn.Module] = []
            is_last = k == len(channels) - 2

            # Set dropout for the input of the conv sequence if defined
            if k == 0 and dropout_input:
                assert 0 < dropout_input < 1
                layers.append(nn.Dropout(dropout_input))

            # conv layer
            if dilation_growth > 1:
                assert kernel % 2 != 0, "Supports only odd kernel with dilation for now"
            if dilation_period and (k % dilation_period) == 0:
                dilation = 1
            pad = kernel // 2 * dilation
            layers.append(Conv(chin, chout, kernel, stride, pad,
                               dilation=dilation, groups=groups if k > 0 else 1))
            dilation *= dilation_growth # dilation_growth = 2
            # non-linearity
            if activation_on_last or not is_last:
                if batch_norm:
                    layers.append(nn.BatchNorm1d(num_features=chout))
                layers.append(activation())
                if dropout:
                    layers.append(nn.Dropout(dropout))
                if rewrite:
                    layers += [nn.Conv1d(chout, chout, 1), nn.LeakyReLU(leakiness)]
                    # layers += [nn.Conv1d(chout, 2 * chout, 1), nn.GLU(dim=1)]
            if chin == chout and skip:
                if scale is not None:
                    layers.append(LayerScale(chout, scale))
                if post_skip:
                    layers.append(Conv(chout, chout, 1, groups=chout, bias=False))

            self.sequence.append(nn.Sequential(*layers))
            if glu and (k + 1) % glu == 0:
                ch = 2 * chout if glu_glu else chout
                act = nn.GLU(dim=1) if glu_glu else activation()
                self.glus.append(
                    nn.Sequential(
                        nn.Conv1d(chout, ch, 1 + 2 * glu_context, padding=glu_context), act))
            else:
                self.glus.append(None)

    def forward(self, x: tp.Any) -> tp.Any:
        for module_idx, module in enumerate(self.sequence):
            old_x = x
            x = module(x)
            if self.skip and x.shape == old_x.shape:
                x = x + old_x
            glu = self.glus[module_idx]
            if glu is not None:
                x = glu(x)
        return x

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
                 hidden: tp.Dict[str, int] = {"meg": 320},
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
                 run_name = 'default',
                 useful_length = 300,
                 ):
        super().__init__()
        self.run_name= run_name
        if set(in_channels.keys()) != set(hidden.keys()):
            raise ValueError("Channels and hidden keys must match "
                             f"({set(in_channels.keys())} and {set(hidden.keys())})")
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
        sizes = {'meg': [320, 320, 320, 320, 320, 320]}
        self.conv_encoder = nn.ModuleDict({name: ConvSequence(channels)
                                           for name, channels in sizes.items()})


        self.embed_positions = PositionalEmbedding(
                useful_length, 320, padding_idx=0
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

        # inputs["meg"] = self.conv_encoder['meg'](inputs["meg"])
        positions = self.embed_positions(torch.zeros((inputs["meg"].shape[0], inputs["meg"].shape[2]), dtype=torch.bool))
        x = inputs["meg"] + positions.permute(0, 2, 1)

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

    def __init__(self, 
                 encoder_layers=4,
                 dropout=0.1,
                 encoder_layerdrop=0.1,
                 freeze_encoder_updates=130000,
                 no_freeze_encoder_layer=None,
                 use_sent_enc_layer=True,
                 unb_enc_layer=-1,
                 layer_norm_first=False,
                 encoder_embed_dim=320,
                 layer_norm_eps=1e-8,
                 encoder_attention_heads=8,
                 encoder_max_relative_position=1200,
                 encoder_ffn_embed_dim=320*4,
                 attention_dropout=0.1,
                 activation_dropout=0.1,
                 activation_fn="gelu",
                 relative_position_embedding=True
                ):
        super().__init__(None)

        self.register_buffer("version", torch.Tensor([3]))
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )
        self.encoder_layerdrop = encoder_layerdrop
        self.freeze_encoder_updates = freeze_encoder_updates
        self.no_freeze_encoder_layer = no_freeze_encoder_layer
        self.num_updates = 0
        export = False

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [self.build_encoder_layer(use_sent_enc_layer, encoder_embed_dim, encoder_ffn_embed_dim, encoder_attention_heads, 
                            dropout, attention_dropout, activation_dropout, activation_fn, layer_norm_first, relative_position_embedding) 
             for i in range(encoder_layers)]
        )
        self.num_layers = len(self.layers)

        self.use_sent_enc_layer = use_sent_enc_layer
        self.unb_enc_layer = unb_enc_layer

        self.layer_norm_first = layer_norm_first
        self.layer_norm = LayerNorm(encoder_embed_dim, eps=layer_norm_eps, export=export)
        
        self.proj = None
        self.relative_position_embedding = relative_position_embedding
        if relative_position_embedding:
            self.pos_emb = RelativePositionalEncoding(encoder_embed_dim//encoder_attention_heads, encoder_max_relative_position)


    def build_encoder_layer(self, 
                            use_sent_enc_layer, 
                            encoder_embed_dim, 
                            encoder_ffn_embed_dim, 
                            encoder_attention_heads, 
                            dropout, attention_dropout, 
                            activation_dropout, 
                            activation_fn, 
                            layer_norm_first, 
                            relative_position_embedding
                           ):
        if use_sent_enc_layer:
            layer = TransformerSentenceEncoderLayer(
                embedding_dim=encoder_embed_dim,
                ffn_embedding_dim=encoder_ffn_embed_dim,
                num_attention_heads=encoder_attention_heads,
                dropout=dropout,
                attention_dropout=attention_dropout,
                activation_dropout=activation_dropout,
                activation_fn=activation_fn,
                layer_norm_first=layer_norm_first,
                has_relative_attention_bias=relative_position_embedding,
            )
        else:
            layer = TransformerEncoderLayer()
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
            ft = self.num_updates <= self.freeze_encoder_updates
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
            x = encoder_in.permute(1, 0, 2)

            encoder_states = []

            if return_all_hiddens:
                encoder_states.append(x)

            ## relative position embedding
            if self.relative_position_embedding:
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

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m

class Postnet(torch.nn.Module):

    def __init__(
        self,
        idim,
        odim,
        n_layers=3,
        n_chans=320,
        n_filts=5,
        use_batch_norm=True,
    ):
        """Initialize postnet module.

        Args:
            idim (int): Dimension of the inputs.
            odim (int): Dimension of the outputs.
            n_layers (int, optional): The number of layers.
            n_filts (int, optional): The number of filter size.
            n_units (int, optional): The number of filter channels.
            use_batch_norm (bool, optional): Whether to use batch normalization..
            dropout_rate (float, optional): Dropout rate..

        """
        super(Postnet, self).__init__()
        self.postnet = torch.nn.ModuleList()
        for layer in range(n_layers - 1):
            ichans = idim if layer == 0 else n_chans
            ochans = odim if layer == n_layers - 1 else n_chans
            
            if use_batch_norm:
                self.postnet += [
                    torch.nn.Sequential(
                        torch.nn.Conv1d(
                            ichans,
                            ochans,
                            n_filts,
                            stride=1,
                            padding=(n_filts - 1) // 2,
                            bias=False,
                        ),
                        torch.nn.BatchNorm1d(ochans),
                        torch.nn.Tanh(),
                    )
                ]
            else:
                self.postnet += [
                    torch.nn.Sequential(
                        torch.nn.Conv1d(
                            ichans,
                            ochans,
                            n_filts,
                            stride=1,
                            padding=(n_filts - 1) // 2,
                            bias=False,
                        ),
                        torch.nn.GELU(),
                    )
                ]
        ichans = n_chans if n_layers != 1 else odim
        if use_batch_norm:
            self.postnet += [
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        ichans,
                        odim,
                        n_filts,
                        stride=1,
                        padding=(n_filts - 1) // 2,
                        bias=False,
                    ),
                    torch.nn.BatchNorm1d(odim),
                )
            ]
        else:
            self.postnet += [
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        ichans,
                        odim,
                        n_filts,
                        stride=1,
                        padding=(n_filts - 1) // 2,
                        bias=False,
                    ),
                )
            ]

    def forward(self, xs):
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of the sequences of padded input tensors (B, idim, Tmax).

        Returns:
            Tensor: Batch of padded output tensor. (B, odim, Tmax).

        """      
        for i in range(len(self.postnet)):
            xs = self.postnet[i](xs)
        return xs

class ConformerEncoder(FairseqEncoder):
    """Conformer Encoder for speech translation based on https://arxiv.org/abs/2005.08100"""

    def __init__(self, 
                 encoder_embed_dim=320,
                 no_scale_embedding=True,
                 dropout=0.1,
                 encoder_ffn_embed_dim=320*4,
                 encoder_attention_heads=4,
                 depthwise_conv_kernel_size=31,
                 attn_type=None,
                 fp16=True,
                 encoder_layers=4,
                ):
        super().__init__(None)
        self.embed_scale = math.sqrt(encoder_embed_dim)
        if no_scale_embedding:
            self.embed_scale = 1.0
        self.pos_enc_type = "abs"
        self.linear = torch.nn.Linear(encoder_embed_dim, encoder_embed_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.conformer_layers = torch.nn.ModuleList(
            [
                ConformerEncoderLayer(
                    embed_dim=encoder_embed_dim,
                    ffn_embed_dim=encoder_ffn_embed_dim,
                    attention_heads=encoder_attention_heads,
                    dropout=dropout,
                    depthwise_conv_kernel_size=depthwise_conv_kernel_size,
                    attn_type=attn_type,
                    pos_enc_type=self.pos_enc_type,
                    use_fp16=fp16,
                )
                for _ in range(encoder_layers)
            ]
        )

    def forward(self, x, encoder_padding_mask, return_all_hiddens=False):
        """
        Args:
            src_tokens: Input source tokens Tensor of shape B X T X C
            src_lengths: Lengths Tensor corresponding to input source tokens
            return_all_hiddens: If true will append the self attention states to the encoder states
        Returns:
            encoder_out: Tensor of shape B X T X C
            encoder_padding_mask: Optional Tensor with mask
            encoder_embedding: Optional Tensor. Always empty here
            encoder_states: List of Optional Tensors wih self attention states
            src_tokens: Optional Tensor. Always empty here
            src_lengths: Optional Tensor. Always empty here
        """
        x = x.permute(1, 0, 2)
        x = self.embed_scale * x
        x = self.linear(x)
        x = self.dropout(x)
        encoder_states = []
        positions = None
        # x is T X B X C
        for layer in self.conformer_layers:
            x, _ = layer(x, encoder_padding_mask, positions)
            if return_all_hiddens:
                encoder_states.append(x)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask]
            if encoder_padding_mask.any()
            else [],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        """Required method for a FairseqEncoder. Calls the method from the parent class"""
        return S2TTransformerEncoder.reorder_encoder_out(self, encoder_out, new_order)

# class CTCLayer(nn.Module):
#     def __init__(
#         self,
#         name: str,
#         input_key: str,
#         target_key: str,
#         blank_idx: int = 0,
#         padding_idx: int = None,
#         input_lengths_key: str = None,
#     ):
#         super().__init__()
#         self.ctc_loss = nn.CTCLoss(blank=blank_idx, reduction='mean', zero_infinity=True)
#         self.input_key = input_key
#         self.target_key = target_key
#         self.input_lengths_key = input_lengths_key
#         self.blank_idx = blank_idx
#         self.padding_idx = padding_idx if padding_idx is not None else blank_idx + 1

#     def forward(self, info):
#         """
#         Computes the CTC loss.

#         Args:
#             info (dict): Dictionary containing model outputs and other relevant data.
#                 - info[self.input_key]: Model logits of shape (batch_size, sequence_length, num_classes).
#                 - info[self.target_key]: Target data (list of dicts with 'phone' key).
#                 - info[self.input_lengths_key]: (Optional) Actual lengths of the input sequences.

#         Returns:
#             loss (Tensor): The computed CTC loss, scaled by the weight.
#         """
#         # Build targets and target lengths
#         padded_targets, target_lengths = build_target(info[self.target_key], self.padding_idx)

#         # Get logits from the model output
#         logits = info[self.input_key]  # Expected shape: (batch_size, sequence_length, num_classes)

#         # Move logits to the device of phonemes
#         device = padded_targets.device
#         logits = logits.to(device)

#         # Apply log_softmax to obtain log probabilities
#         log_probs = F.log_softmax(logits, dim=-1)  # Shape: (batch_size, seq_length, num_classes)

#         # Transpose log_probs to match (seq_length, batch_size, num_classes)
#         log_probs = log_probs.permute(1, 0, 2)  # Now shape is (seq_length, batch_size, num_classes)

#         # Determine input lengths
#         if self.input_lengths_key and self.input_lengths_key in info:
#             input_lengths = info[self.input_lengths_key].to(device)
#         else:
#             # Assume all input sequences have the same length
#             input_lengths = torch.full(
#                 (log_probs.size(1),),  # batch_size
#                 log_probs.size(0),     # seq_length
#                 dtype=torch.long,
#                 device=device
#             )

#         # Compute the CTC loss
#         loss = self.ctc_loss(log_probs, padded_targets, input_lengths, target_lengths)

#         loss = self.weight * loss

#         return loss

class CTCLayer(nn.Module):
    def __init__(self,
                 # Channels
                 in_dimension: int,
                 vocab_size: int
                 ):
        super().__init__()
        # self.encoder = EncoderStableLayerNorm(in_dimension)
        self.linear = nn.Linear(in_dimension, vocab_size)


    def forward(self, inputs):
        # encoder_outputs = self.encoder(inputs.transpose(1,2))
    
        # inputs = self.dropout(encoder_outputs)
        inputs = self.linear(inputs.transpose(1,2))
        out = nn.functional.log_softmax(inputs, dim=-1, dtype=torch.float32).transpose(0, 1)

        return out




class BrainTransformer(nn.Module):
    def __init__(
        self,
        in_channels: dict,
        run_name: str = 'test',
        depth: int = 4,
        useful_length: int = 300,
    ) -> None:

        super().__init__()
        self.pre_layer = ConvPreNet(in_channels=in_channels, run_name=run_name, useful_length=useful_length)
        self.transformer = TransformerEncoder(encoder_layers=depth)
        # self.post_layer = Postnet(idim=320,odim=80)
        self.post_layer = nn.Sequential(
                nn.Conv1d(320, 2 * 320, 1),
                nn.GELU(),
                nn.ConvTranspose1d(320 * 2, 80, kernel_size=1, stride=1, padding=0))
        self.ctc_layer = CTCLayer(320, 32)


    def forward(
        self,
        inputs: dict, 
        batch: dict,
    ):
        x = self.pre_layer(inputs=inputs, batch=batch).permute(0, 2, 1)
        x = self.transformer(x, (torch.arange(inputs["meg"].shape[2]).expand(inputs["meg"].shape[0], -1) >= 1000).to('cuda'))
        out = self.post_layer(x["encoder_out"][0].permute(1, 2, 0))
        ctc_out = self.ctc_layer(x["encoder_out"][0].permute(1, 2, 0))
        return out, ctc_out


