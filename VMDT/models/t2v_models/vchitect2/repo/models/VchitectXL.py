# Copyright 2024 Stability AI and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Any, Dict, Optional, Union, Tuple, List

import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from .blocks import JointTransformerBlock
# from diffusers.models.attention_processor import Attention, AttentionProcessor
from .attention import Attention, AttentionProcessor
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.embeddings import CombinedTimestepTextProjEmbeddings, PatchEmbed
from diffusers.models.transformers.transformer_2d import Transformer2DModelOutput

from einops import rearrange
from torch.distributed._tensor import Shard, Replicate
from torch.distributed.tensor.parallel import (
    parallelize_module,
    PrepareModuleOutput
)

#from models.layers import ParallelTimestepEmbedder, TransformerBlock, ParallelFinalLayer, Identity


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class VchitectXLTransformerModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    """
    The Transformer model introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        sample_size (`int`): The width of the latent images. This is fixed during training since
            it is used to learn a number of position embeddings.
        patch_size (`int`): Patch size to turn the input data into small patches.
        in_channels (`int`, *optional*, defaults to 16): The number of channels in the input.
        num_layers (`int`, *optional*, defaults to 18): The number of layers of Transformer blocks to use.
        attention_head_dim (`int`, *optional*, defaults to 64): The number of channels in each head.
        num_attention_heads (`int`, *optional*, defaults to 18): The number of heads to use for multi-head attention.
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        caption_projection_dim (`int`): Number of dimensions to use when projecting the `encoder_hidden_states`.
        pooled_projection_dim (`int`): Number of dimensions to use when projecting the `pooled_projections`.
        out_channels (`int`, defaults to 16): Number of output channels.

    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: int = 128,
        patch_size: int = 2,
        in_channels: int = 16,
        num_layers: int = 18,
        attention_head_dim: int = 64,
        num_attention_heads: int = 18,
        joint_attention_dim: int = 4096,
        caption_projection_dim: int = 1152,
        pooled_projection_dim: int = 2048,
        out_channels: int = 16,
        pos_embed_max_size: int = 96,
        tp_size: int = 1,
        rope_scaling_factor: float = 1.,
    ):
        super().__init__()
        default_out_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else default_out_channels
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim

        self.pos_embed = PatchEmbed(
            height=self.config.sample_size,
            width=self.config.sample_size,
            patch_size=self.config.patch_size,
            in_channels=self.config.in_channels,
            embed_dim=self.inner_dim,
            pos_embed_max_size=pos_embed_max_size,  # hard-code for now.
        )
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=self.inner_dim, pooled_projection_dim=self.config.pooled_projection_dim
        )
        self.context_embedder = nn.Linear(self.config.joint_attention_dim, self.config.caption_projection_dim)
        # `attention_head_dim` is doubled to account for the mixing.
        # It needs to crafted when we get the actual checkpoints.
        self.transformer_blocks = nn.ModuleList(
            [
                JointTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.inner_dim,
                    context_pre_only=i == num_layers - 1,
                    tp_size = tp_size
                )
                for i in range(self.config.num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False

        # Video param
        # self.scatter_dim_zero = Identity()
        self.freqs_cis = VchitectXLTransformerModel.precompute_freqs_cis(
            self.inner_dim // self.config.num_attention_heads, 1000000, theta=1e6, rope_scaling_factor=rope_scaling_factor  # todo max pos embeds
        )

        #self.vid_token = nn.Parameter(torch.empty(self.inner_dim))

    @staticmethod
    def tp_parallelize(model, tp_mesh):
        for layer_id, transformer_block in enumerate(model.transformer_blocks):
            layer_tp_plan = {
                # Attention layer
                "attn.gather_seq_scatter_hidden": PrepareModuleOutput(
                    output_layouts=Replicate(),
                    desired_output_layouts=Shard(-2)
                ),
                "attn.gather_hidden_scatter_seq": PrepareModuleOutput(
                    output_layouts=Shard(-2),
                    desired_output_layouts=Replicate(),
                )
            }
            parallelize_module(
                module=transformer_block,
                device_mesh=tp_mesh,
                parallelize_plan=layer_tp_plan
            )
        return model

    @staticmethod
    def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, rope_scaling_factor: float = 1.0):
        freqs = 1.0 / (theta ** (
                torch.arange(0, dim, 2)[: (dim // 2)].float() / dim
        ))
        t = torch.arange(end, device=freqs.device, dtype=torch.float)
        t = t / rope_scaling_factor
        freqs = torch.outer(t, freqs).float()  
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        return freqs_cis
    
    # Copied from diffusers.models.unets.unet_3d_condition.UNet3DConditionModel.enable_forward_chunking
    def enable_forward_chunking(self, chunk_size: Optional[int] = None, dim: int = 0) -> None:
        """
        Sets the attention processor to use [feed forward
        chunking](https://huggingface.co/blog/reformer#2-chunked-feed-forward-layers).

        Parameters:
            chunk_size (`int`, *optional*):
                The chunk size of the feed-forward layers. If not specified, will run feed-forward layer individually
                over each tensor of dim=`dim`.
            dim (`int`, *optional*, defaults to `0`):
                The dimension over which the feed-forward computation should be chunked. Choose between dim=0 (batch)
                or dim=1 (sequence length).
        """
        if dim not in [0, 1]:
            raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")

        # By default chunk size is 1
        chunk_size = chunk_size or 1

        def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, chunk_size, dim)

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor(return_deprecated_lora=True)

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def patchify_and_embed(self, x):
        pH = pW = self.patch_size
        B, F, C, H, W = x.size()
        x = rearrange(x, "b f c h w -> (b f) c h w")
        x = self.pos_embed(x) # [B L D]
        # x = torch.cat([
        #     x,
        #     self.vid_token.view(1, 1, -1).expand(B*F, 1, -1),
        # ], dim=1) 

        return x, F, [(H, W)] * B
    
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The [`VchitectXLTransformerModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        # if USE_PEFT_BACKEND:
        #     # weight the lora layers by setting `lora_scale` for each PEFT layer
        #     scale_lora_layers(self, lora_scale)
        # else:
        #     logger.warning(
        #         "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
        #     )

        height, width = hidden_states.shape[-2:]

        batch_size = hidden_states.shape[0]
        hidden_states, F_num, _ = self.patchify_and_embed(hidden_states)  # takes care of adding positional embeddings too.
        full_seq = batch_size * F_num

        self.freqs_cis = self.freqs_cis.to(hidden_states.device)
        freqs_cis = self.freqs_cis
        # seq_length = hidden_states.size(1)
        # freqs_cis = self.freqs_cis[:hidden_states.size(1)*F_num]
        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        # for block in self.transformer_blocks:
        #     if self.training and self.gradient_checkpointing:

        #         def create_custom_forward(module, return_dict=None):
        #             def custom_forward(*inputs):
        #                 if return_dict is not None:
        #                     return module(*inputs, return_dict=return_dict)
        #                 else:
        #                     return module(*inputs)

        #             return custom_forward

        #         ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
        #         hidden_states = torch.utils.checkpoint.checkpoint(
        #             create_custom_forward(block),
        #             hidden_states,
        #             encoder_hidden_states,
        #             temb,
        #             **ckpt_kwargs,
        #         )

        #     else:
        #         encoder_hidden_states, hidden_states = block(
        #             hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb
        #         )

        for block_idx, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb.repeat(F_num,1), freqs_cis=freqs_cis, full_seqlen=full_seq, Frame=F_num
            )
            
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        # unpatchify
        # hidden_states = hidden_states[:, :-1] #Drop the video token

        # unpatchify
        patch_size = self.config.patch_size
        height = height // patch_size
        width = width // patch_size

        hidden_states = hidden_states.reshape(
            shape=(hidden_states.shape[0], height, width, patch_size, patch_size, self.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(hidden_states.shape[0], self.out_channels, height * patch_size, width * patch_size)
        )

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        return list(self.transformer_blocks)
    
    @classmethod
    def from_pretrained_temporal(cls, pretrained_model_path, torch_dtype, logger, subfolder=None, tp_size=1):

        import os
        import json

        if subfolder is not None:
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)

        config_file = os.path.join(pretrained_model_path, 'config.json')

        with open(config_file, "r") as f:
            config = json.load(f)

        config["tp_size"] = tp_size
        from diffusers.utils import WEIGHTS_NAME
        from safetensors.torch import load_file,load_model
        model = cls.from_config(config)
        # model_file = os.path.join(pretrained_model_path, WEIGHTS_NAME)
    
        model_files = [
            os.path.join(pretrained_model_path, 'diffusion_pytorch_model.bin'),
            os.path.join(pretrained_model_path, 'diffusion_pytorch_model.safetensors')
        ]

        model_file = None

        for fp in model_files:
            if os.path.exists(fp):
                model_file = fp

        if not model_file:
            raise RuntimeError(f"{model_file} does not exist")

        if not os.path.isfile(model_file):
            raise RuntimeError(f"{model_file} does not exist")
        

        state_dict = load_file(model_file,device="cpu")
        m, u = model.load_state_dict(state_dict, strict=False)
        model = model.to(torch_dtype)

        params = [p.numel() if "temporal" in n else 0 for n, p in model.named_parameters()]
        total_params = [p.numel() for n, p in model.named_parameters()]

        if logger is not None:
            logger.info(f"model_file: {model_file}")
            logger.info(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")
            logger.info(f"### Temporal Module Parameters: {sum(params) / 1e6} M")
            logger.info(f"### Total Parameters: {sum(total_params) / 1e6} M")
        
        return model