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
from einops import rearrange

from conv import CausalConv3d, CausalGroupNorm

class CausalResnet3d(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dropout: float = 0.0,
                 eps: float = 1e-5,
                 scale_factor: float = 1.0,
                 norm_num_groups: int = 32,
                 ):
        
        """ 
            in_channels (`int`): input channels 
            out_channels (`int`): output channels 
            dropout (`float`): what is the dropout to put on the `nn.GroupNorm()`
            eps (`float`): what is the eps to put on the `nn.GroupNorm()`
            scale_factor (`float`): the float value to scale the video tensor 
            norm_num_groups (`int`): the value to use in the `nn.GroupNorm()`
        """
        super().__init__()
        self.scale_factor = scale_factor
        
        self.norm1 = CausalGroupNorm(in_channels=in_channels,
                                     num_groups=norm_num_groups,
                                     eps=eps)
        self.conv1 = CausalConv3d(in_channels=in_channels,
                                  out_channels=out_channels)
        
        self.norm2 = CausalGroupNorm(in_channels=out_channels,
                                     num_groups=norm_num_groups,
                                     eps=eps)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = CausalConv3d(in_channels=out_channels,
                                  out_channels=out_channels)
        self.act_fn = nn.SiLU()

        output_channels = out_channels



        # [128] != [256]
        self.in_out_not_equal = None
        if in_channels != output_channels:
            self.in_out_not_equal = CausalConv3d(in_channels=in_channels,
                                              out_channels=output_channels,
                                              kernel_size=1,
                                              stride=1,
                                              bias=True
                                              )

        
       

        
            
    def forward(self, x):

        # feed-forward tensor layer
        fd_tensor = x
       

        x = self.norm1(x)
        x = self.act_fn(x)
        x = self.conv1(x)
        
        x = self.norm2(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.conv2(x)


        # [128] != [256]
        # [256] != [512]
        if self.in_out_not_equal is not None:
            # [2, 128, 8, 256, 256] -> [2, 256, 8, 256, 256]
            # [2, 256, 8, 256, 256] -> [2, 512, 8, 256, 256]
            fd_tensor = self.in_out_not_equal(fd_tensor)

        # [2, 128, 8, 256, 256] + [2, 128, 8, 256, 256]
        # [2, 128, 8, 256, 256] + [2, 128, 8, 256, 256]
        # [2, 256, 8, 256, 256] + [2, 256, 8, 256, 256]
        # [2, 256, 8, 256, 256] + [2, 256, 8, 256, 256]
        # [2, 512, 8, 256, 256] + [2, 512, 8, 256, 256]
        x = (fd_tensor + x) / self.scale_factor

        return x 
    

class CausalUpsampleHeigthWidth(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super().__init__()

        # [512] -> [4*256]
        self.conv = CausalConv3d(in_channels=in_channels,
                                 out_channels=out_channels*4,
                                 kernel_size=3,
                                 stride=1,
                                 bias=True)
        

    def forward(self,
                x):
        
        x = self.conv(x)
        # [2, (256*2*2), 8, 256, 256] -> [2, 256, 8, (2*256), (2*256)]
        x = rearrange(x, 'b (c p1 p2) t h w -> b c t (h p1) (w p2)', 
                      p1=2, p2=2)
        
        return x 
    

class CausalTemporalUpsample2x(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 ):
        super().__init__()

        self.conv = CausalConv3d(in_channels=in_channels,
                                 out_channels=out_channels * 2,
                                 kernel_size=3,
                                 stride=1,
                                 bias=True)
        
    def forward(self,
                x):
        
        b, c, t, h, w = x.shape 
        x = self.conv(x)
        x = rearrange(x,
                      'b (c p) t h w -> b c (t p) h w', t=t, p=2)
        # [2, (256*2), 8, 256, 256] -> [2, 512, (8*2-1), 256, 256]
        x = x[:, :, 1:]
        return x 


    
