# from mmpretrain.apis import get_model,list_models
# from mmengine import Config
# import torch
# config=Config(dict(model=dict(
#     type='ImageClassifier',
#     backbone=dict(
#         type='TransFG',
#         arch='b',
#         img_size=56,
#         patch_size=16,
#         drop_rate=0.1,
#         init_cfg=[
#             dict(
#                 type='Kaiming',
#                 layer='Conv2d',
#                 mode='fan_in',
#                 nonlinearity='linear')
#         ],
#         patch_cfg=dict(stride=12,padding=0)
#         ),
#     neck=None,
#     head=dict(
#         type='TransFGClsHead',
#         num_classes=10,
#         in_channels=768,
#         loss=dict(
#             type='TransFGLoss'
#         )
#     )))
#     )
# model=get_model(config)
# torch.random.manual_seed(22)
# input=torch.rand((1,3,56,56))
# output=model(input)
# print(model)
# print(output[0],output[0].shape)
# print(output[1],output[1].shape)
# print(list_models())
import torch
from colt5_attention.vit import ConditionalRoutedViT

vit = ConditionalRoutedViT(
    image_size = 256,                # image size
    patch_size = 32,                 # patch size
    num_classes = 1000,              # number of output classes
    dim = 1024,                      # feature dimension
    depth = 6,                       # depth
    attn_num_heavy_tokens_q = 16,    # number of routed queries for heavy attention
    attn_num_heavy_tokens_kv = 16,   # number of routed key/values for heavy attention
    attn_heavy_dim_head = 64,        # dimension per attention head for heavy
    attn_heavy_heads = 8,            # number of attention heads for heavy
    attn_light_window_size = 4,      # the local windowed attention for light branch
    attn_light_dim_head = 32,        # dimension per head for local light attention
    attn_light_heads = 4,            # number of attention heads for local windowed attention
    ff_num_heavy_tokens = 16,        # number of tokens routed for heavy feedforward
    ff_heavy_mult = 4,               # the expansion factor of the heavy feedforward branch
    ff_light_mult = 2,               # expansion factor of the light feedforward branch
    router_use_triton=True
)

images = torch.randn(1, 3, 256, 256)

vit(images)