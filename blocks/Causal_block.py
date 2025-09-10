# Copyright 2025 The savitri-AI Team. All rights reserved.
#
#    company: https://github.com/savitri-AI
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
import torch
from torch import nn 
from diffusers.models.attention_processor import Attention
from einops import rearrange

from resnets.Causal_resnet import (
    CausalResnet3d,
    CausalUpsampleHeigthWidth,
    CausalTemporalUpsample2x
)





class CausalMiddleBlock3d(nn.Module):

    def __init__(self,
                in_channels: int,
                attention_head_dim: int = 512,
                norm_num_groups: int=32,
                dropout: bool = 0.0,
                scale_factor: bool = 1.0,
                eps: float = 1e-5,
                ):
        super().__init__()

        
        resnets = [
            CausalResnet3d(
                in_channels=in_channels,
                out_channels=in_channels,
                norm_num_groups=norm_num_groups,
                dropout=dropout,
                scale_factor=scale_factor,
                eps=eps
            )
        ]

        attentions = []
        for _ in range(1):
            attentions.append(
                Attention(query_dim=in_channels,
                          heads=in_channels // attention_head_dim,
                          dim_head=attention_head_dim,
                          rescale_output_factor=scale_factor,
                          eps=eps,
                          norm_num_groups=norm_num_groups,
                          spatial_norm_dim=None,
                          residual_connection=True,
                          bias=True,
                          upcast_softmax=True,
                          _from_deprecated_attn_block=True)
            )

        resnets.append(
            CausalResnet3d(
                in_channels=in_channels,
                out_channels=in_channels,
                norm_num_groups=norm_num_groups,
                dropout=dropout,
                scale_factor=scale_factor,
                eps=eps

            )
        )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self,
                x):
        
        x = self.resnets[0](x)
        b, c, t, h, w = x.shape

        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            x = rearrange(x, 'b c t h w -> (b t) c h w')
            x = attn(x)
            x = rearrange(x, '(b t) c h w -> b c t h w', t=t)

        x = resnet(x)

        return x 
    


class CausalUpperBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_layers: int = 3,
                 norm_num_groups: int = 32,
                 add_height_width_2x: bool = True,
                 add_frame_2x: bool = True,
                 dropout: float = 0.0,
                 eps: float = 1e-5,
                 scale_factor: float = 1.0):
        super().__init__()
        self.add_height_width_2x = add_height_width_2x
        self.add_frame_2x = add_frame_2x

        self.resnets = nn.ModuleList([])
        for i in range(num_layers):
             input_channels = in_channels if i==0 else out_channels

             self.resnets.append(
                 CausalResnet3d(in_channels=input_channels,
                                out_channels=out_channels,
                                dropout=dropout,
                                eps=eps,
                                scale_factor=scale_factor,
                                norm_num_groups=norm_num_groups,
                                )
             )


        if add_height_width_2x:
            self.upsamplers_height_width = nn.ModuleList([
                CausalUpsampleHeigthWidth(in_channels=out_channels,
                                          out_channels=out_channels)
            ])

        if add_frame_2x:
            self.upsamplers_frame = nn.ModuleList([
                CausalTemporalUpsample2x(in_channels=out_channels,
                                         out_channels=out_channels)
            ])

    
    def forward(self,
                x):
        
        for resnet in self.resnets:
            x = resnet(x)

        if self.add_height_width_2x:
            for upsampler_height_width in self.upsamplers_height_width:
                x = upsampler_height_width(x)

        if self.add_frame_2x:
            for upsampler_frame in self.upsamplers_frame:
                x = upsampler_frame(x)

        return x 
    
    

