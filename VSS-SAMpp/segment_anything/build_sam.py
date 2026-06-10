import torch
from torch.nn import functional as F
from functools import partial

from .modeling import ImageEncoderViT, Sam, Decoder, CenterDecoder, VMAmba


def build_sam_vit_l(
    image_size,
    num_classes,
    dropout_p,
    pixel_mean=[123.675, 116.28, 103.53],
    pixel_std=[58.395, 57.12, 57.375],
    checkpoint=None,
):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        mamba_encoder_embed_dim=384,
        mamba_encoder_depth=12,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
        num_classes=num_classes,
        image_size=image_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
        dropout_p=dropout_p,
        hidden_states=[5, 11, 17, 23],
        mamba_hidden_states=[2, 5, 8, 11],
    )


def _unsupported_vit_builder(model_type):
    def _builder(*args, **kwargs):
        raise ValueError(
            f"Unsupported model_type='{model_type}'. "
            "VSS-SAM++ currently supports only 'vit_l'. "
            "Please set model_type='vit_l' or model_type='default', "
            "and use a ViT-L checkpoint."
        )
    return _builder


# Default model: ViT-L only
build_sam = build_sam_vit_l


sam_model_registry = {
    "default": build_sam_vit_l,
    "vit_l": build_sam_vit_l,
    "vit_h": _unsupported_vit_builder("vit_h"),
    "vit_b": _unsupported_vit_builder("vit_b"),
}


def _build_sam(
        encoder_embed_dim,
        encoder_depth,
        mamba_encoder_embed_dim,
        mamba_encoder_depth,
        encoder_num_heads,
        encoder_global_attn_indexes,
        num_classes,
        image_size,
        pixel_mean,
        pixel_std,
        dropout_p,
        hidden_states,
        mamba_hidden_states,
        checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = image_size
    vit_patch_size = 16
    base_channels = 64
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        vmamba=VMAmba(
            depth=mamba_encoder_depth,
            embed_dim=mamba_encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            patch_size=vit_patch_size,
            d_size = 5

        ),
        Decoder=Decoder(
            in_channels=encoder_embed_dim,
            base_channels=base_channels,
            out_channels=num_classes + 1,
            dropout_p=dropout_p,
            hidden_states=hidden_states,
            trilinear=True
        ),
        seg_decoder=CenterDecoder(
            hidden_states=mamba_hidden_states,
        ),

        pixel_mean=pixel_mean,
        pixel_std=pixel_std
    )

    sam.train()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        try:
            sam.load_state_dict(state_dict)
        except:
            new_state_dict = load_from(sam, state_dict, image_size, vit_patch_size, encoder_global_attn_indexes)
            sam.load_state_dict(new_state_dict)
    return sam, image_embedding_size


def load_from(sam, state_dict, image_size, vit_patch_size, encoder_global_attn_indexes):
    ega = encoder_global_attn_indexes
    sam_dict = sam.state_dict()
    except_keys = ['mask_tokens', 'output_hypernetworks_mlps', 'iou_prediction_head']
    new_state_dict = {k: v for k, v in state_dict.items() if
                      k in sam_dict.keys() and except_keys[0] not in k and except_keys[1] not in k and except_keys[2] not in k}
    pos_embed = new_state_dict['image_encoder.pos_embed']
    token_size = int(image_size // vit_patch_size)
    if pos_embed.shape[1] != token_size:
        
        pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
        pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
        new_state_dict['image_encoder.pos_embed'] = pos_embed
        rel_pos_keys = [k for k in sam_dict.keys() if 'rel_pos' in k]
        global_rel_pos_keys = []
        for rel_pos_key in rel_pos_keys:
            num = int(rel_pos_key.split('.')[2])
            if num in encoder_global_attn_indexes:
                global_rel_pos_keys.append(rel_pos_key)
        
        for k in global_rel_pos_keys:
            rel_pos_params = new_state_dict[k]
            h, w = rel_pos_params.shape
            rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
            rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
            new_state_dict[k] = rel_pos_params[0, 0, ...]
    sam_dict.update(new_state_dict)
    return sam_dict
