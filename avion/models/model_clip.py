import clip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from avion.models.transformer import TextTransformer, VisionTransformer
from avion.models.utils import enable_grad_checkpointing, remap_keys_from_open_clip_to_vit

from avion.models.efficientvit_cls import ClsHead
from efficientvit.models.utils import build_kwargs_from_config

from efficientvit.models.nn import (
    ConvLayer,
    DSConv,
    EfficientViTBlock,
    FusedMBConv,
    IdentityLayer,
    MBConv,
    OpSequential,
    ResBlock,
    ResidualBlock,
)

class VideoClassifier(nn.Module):
    def __init__(self,
                 vision_model: nn.Module,
                 dropout: float,
                 num_classes: int,
                 **kwargs,
                 ):
        super().__init__()
        #print(f"In VideoClassifier, vidion_model = {vision_model}")
        self.visual = vision_model
        self.dropout = nn.Dropout(dropout)
        if hasattr(self.visual, "image_projection"):
            self.visual.image_projection = None
        #self.fc_cls = nn.Linear(vision_model.width, num_classes, bias=True)
        #self.fc_cls.weight.data.normal_(mean=0.0, std=0.01)
        #self.fc_cls.bias.data.zero_()
        self.fc_cls = ClsHead(       
                in_channels=128,
                width_list=[1024, 1280],
                n_classes=3806,
                dropout=0.0,
                norm="bn2d",
                act_func="hswish",
                fid="stage_final",
                **build_kwargs_from_config(kwargs, ClsHead),
            )

    def forward(self, image):
        image_embed = self.visual(image)
        if isinstance(image_embed, list):
            assert len(image_embed) == 1
            image_embed = image_embed[0]
        logit = self.fc_cls(image_embed)
        return logit
    

class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 vision_model: nn.Module,
                 text_model: nn.Module,
                 vision_width: int = None,
                 text_width: int = None,
                 freeze_temperature=False,
                 width_list=[8, 16, 32, 64, 128],
                 depth_list=[1, 2, 2, 2, 2],
                 dim=16,
                 expand_ratio=4,
                 norm="bn2d",
                 act_func="hswish",
                 **kwargs
    ):
        super().__init__()

        self.visual = vision_model
        self.textual = text_model

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        if freeze_temperature:
            self.logit_scale.requires_grad_(False)
        if vision_width is not None:
            self.vision_width = vision_width
            self.image_projection = nn.Parameter(torch.empty(vision_width, embed_dim))
        else:
            self.image_projection = None
        if text_width is not None:
            self.text_width = text_width
            self.text_projection = nn.Parameter(torch.empty(text_width, embed_dim))
        else:
            self.text_projection = None

        self.init_parameters()

        #ported from EfficientViTBackbone
        self.width_list = []
        # input stem
        self.input_stem = [
            ConvLayer(
                in_channels=3,
                out_channels=width_list[0],
                stride=2,
                norm=norm,
                act_func=act_func,
            )
        ]
        for _ in range(depth_list[0]):
            block = self.build_local_block(
                in_channels=width_list[0],
                out_channels=width_list[0],
                stride=1,
                expand_ratio=1,
                norm=norm,
                act_func=act_func,
            )
            self.input_stem.append(ResidualBlock(block, IdentityLayer()))
        in_channels = width_list[0]
        self.input_stem = OpSequential(self.input_stem)
        self.width_list.append(in_channels)
        self.width = self.width_list[0]
        # stages
        self.stages = []
        for w, d in zip(width_list[1:3], depth_list[1:3]):
            stage = []
            for i in range(d):
                stride = 2 if i == 0 else 1
                block = self.build_local_block(
                    in_channels=in_channels,
                    out_channels=w,
                    stride=stride,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=act_func,
                )
                block = ResidualBlock(block, IdentityLayer() if stride == 1 else None)
                stage.append(block)
                in_channels = w
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)

        for w, d in zip(width_list[3:], depth_list[3:]):
            stage = []
            block = self.build_local_block(
                in_channels=in_channels,
                out_channels=w,
                stride=2,
                expand_ratio=expand_ratio,
                norm=norm,
                act_func=act_func,
                fewer_norm=True,
            )
            stage.append(ResidualBlock(block, None))
            in_channels = w

            for _ in range(d):
                stage.append(
                    EfficientViTBlock(
                        in_channels=in_channels,
                        dim=dim,
                        expand_ratio=expand_ratio,
                        norm=norm,
                        act_func=act_func,
                    )
                )
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)
        self.stages = nn.ModuleList(self.stages)
    @staticmethod
    def build_local_block(
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: float,
        norm: str,
        act_func: str,
        fewer_norm: bool = False,
    ) -> nn.Module:
        if expand_ratio == 1:
            block = DSConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
            )
        else:
            block = MBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, True, False) if fewer_norm else False,
                norm=(None, None, norm) if fewer_norm else norm,
                act_func=(act_func, act_func, None),
            )
        return block


    def init_parameters(self):
        if self.image_projection is not None:
            trunc_normal_(self.image_projection, std=self.vision_width ** -0.5)
        if self.text_projection is not None:
            trunc_normal_(self.text_projection, std=self.text_width ** -0.5)

    def encode_image(self, image):
        x = self.visual(image)
        if self.image_projection is not None:
            x = x @ self.image_projection.to(x.dtype)
        return x

    def encode_text(self, text, cast_dtype=None):
        x = self.textual(text, cast_dtype=cast_dtype)
        if self.text_projection is not None:
            x = x @ self.text_projection.to(x.dtype)
        return x

    def forward(self, image, text):
        image_embed = self.encode_image(image)
        text_embed = self.encode_text(text, cast_dtype=image_embed.dtype)

        return F.normalize(image_embed, dim=-1), F.normalize(text_embed, dim=-1), self.logit_scale.exp()

def CLIP_VITB16(
    freeze_temperature=False,
    use_grad_checkpointing=False,
    use_bidirectional_lm=False,
    context_length=77,
    patch_dropout=0.,
    drop_path_rate=0.,
    num_frames=1,
    use_fast_conv1=False,
    use_flash_attn=False,
    project_embed_dim=512,
    pretrain_zoo='openai',
    pretrain_path=None,
    **kwargs
):
    # vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
    vision_model = VisionTransformer(
        224, 16, 768, 12, 12, 4,
        output_dim=project_embed_dim, patch_dropout=patch_dropout,
        drop_path_rate=drop_path_rate,
        num_frames=num_frames,
        use_fast_conv1=use_fast_conv1,
        use_flash_attn=use_flash_attn,
    )
    text_model = TextTransformer(context_length=context_length, vocab_size=49408, width=512, heads=8, layers=12, output_dim=project_embed_dim, causal_mask=not use_bidirectional_lm)
    enable_grad_checkpointing(vision_model, use_grad_checkpointing)
    enable_grad_checkpointing(text_model, use_grad_checkpointing)
    model = CLIP(embed_dim=project_embed_dim, vision_model=vision_model, text_model=text_model, freeze_temperature=freeze_temperature)

    if pretrain_zoo == "openai":
        print("=> loading openai model")
        clip_model, _ = clip.load('ViT-B/16', device='cpu')
        remapped_state_dict = remap_keys_from_open_clip_to_vit(
            clip_model.state_dict(),
            use_fast_conv1=use_fast_conv1,
            use_flash_attn=use_flash_attn,
        )
        missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)
    elif pretrain_zoo == "open_clip":
        assert pretrain_path is not None
        state_dict = torch.load(pretrain_path)
        print("=> loading open_clip model")
        remapped_state_dict = remap_keys_from_open_clip_to_vit(state_dict, use_fast_conv1=use_fast_conv1, use_flash_attn=use_flash_attn)
        missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)
    else:
        raise NotImplementedError
    return model


def CLIP_VITL14(
    freeze_temperature=False,
    use_grad_checkpointing=False,
    use_bidirectional_lm=False,
    context_length=77,
    vocab_size=49408,
    patch_dropout=0.,
    drop_path_rate=0.,
    num_frames=1,
    use_fast_conv1=False,
    use_flash_attn=False,
    project_embed_dim=512,
    pretrain_zoo='openai',
    pretrain_path=None,
    **kwargs
):
    # vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
    vision_model = VisionTransformer(
        224, 14, 1024, 24, 16, 4,
        output_dim=project_embed_dim, patch_dropout=patch_dropout,
        drop_path_rate=drop_path_rate,
        num_frames=num_frames,
        use_fast_conv1=use_fast_conv1,
        use_flash_attn=use_flash_attn,
    )
    text_model = TextTransformer(context_length=context_length, vocab_size=vocab_size, width=768, heads=12, layers=12, output_dim=project_embed_dim, causal_mask=not use_bidirectional_lm)
    enable_grad_checkpointing(vision_model, use_grad_checkpointing)
    enable_grad_checkpointing(text_model, use_grad_checkpointing)
    model = CLIP(embed_dim=project_embed_dim, vision_model=vision_model, text_model=text_model, freeze_temperature=freeze_temperature)

    if pretrain_zoo == "openai":
        print("=> loading openai model")
        clip_model, _ = clip.load('ViT-L/14', device='cpu')
        remapped_state_dict = remap_keys_from_open_clip_to_vit(
            clip_model.state_dict(), 24,
            context_length=context_length,
            vocab_size=vocab_size,
            use_fast_conv1=use_fast_conv1,
            use_flash_attn=use_flash_attn,
        )
        missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)
    elif pretrain_zoo == "open_clip":
        assert pretrain_path is not None
        state_dict = torch.load(pretrain_path)
        print("=> loading open_clip model")
        remapped_state_dict = remap_keys_from_open_clip_to_vit(state_dict, use_flash_attn=use_flash_attn)
        missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)
    else:
        raise NotImplementedError
    return model


def CLIP_VITL14_336PX(
    freeze_temperature=False,
    use_grad_checkpointing=False,
    use_bidirectional_lm=False,
    context_length=77,
    vocab_size=49408,
    patch_dropout=0.,
    drop_path_rate=0.,
    num_frames=1,
    use_fast_conv1=False,
    use_flash_attn=False,
    project_embed_dim=512,
    pretrain_zoo='openai',
    pretrain_path=None,
    **kwargs
):
    # vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
    vision_model = VisionTransformer(
        336, 14, 1024, 24, 16, 4,
        output_dim=project_embed_dim, patch_dropout=patch_dropout,
        drop_path_rate=drop_path_rate,
        num_frames=num_frames,
        use_fast_conv1=use_fast_conv1,
        use_flash_attn=use_flash_attn,
    )
    text_model = TextTransformer(context_length=context_length, vocab_size=vocab_size, width=768, heads=12, layers=12, output_dim=project_embed_dim, causal_mask=not use_bidirectional_lm)
    enable_grad_checkpointing(vision_model, use_grad_checkpointing)
    enable_grad_checkpointing(text_model, use_grad_checkpointing)
    model = CLIP(embed_dim=project_embed_dim, vision_model=vision_model, text_model=text_model, freeze_temperature=freeze_temperature)

    if pretrain_zoo == "openai":
        print("=> loading openai model")
        clip_model, _ = clip.load('ViT-L/14@336px', device='cpu')
        remapped_state_dict = remap_keys_from_open_clip_to_vit(
            clip_model.state_dict(), 24,
            context_length=context_length,
            vocab_size=vocab_size,
            use_fast_conv1=use_fast_conv1,
            use_flash_attn=use_flash_attn,
        )
        missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)
    elif pretrain_zoo == "open_clip":
        assert pretrain_path is not None
        state_dict = torch.load(pretrain_path)
        print("=> loading open_clip model")
        remapped_state_dict = remap_keys_from_open_clip_to_vit(state_dict, use_flash_attn=use_flash_attn)
        missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)
    else:
        raise NotImplementedError
    return model
