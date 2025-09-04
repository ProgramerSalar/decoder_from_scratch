import torch 
from torch import nn 
from typing import Union
from timm.layers import trunc_normal_
from einops import rearrange

class CausalConv3d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size: Union[int, tuple[int, int, int]] = 3,
                 stride: Union[int, tuple[int, int, int]] = 1,
                 padding = 0,
                 pad_mode = 'constant',
                 **kwargs
                 ):
        
        super().__init__()
        self.padding = padding
        self.dilation = kwargs.pop('dilation', 1)
        self.pad_mode = pad_mode

        kernel_size = (kernel_size,) * 3 if isinstance(kernel_size, int) else kernel_size
        stride = (stride,)*3 if isinstance(stride, int) else stride
        self.time_kernel_size, self.height_kernel_size, self.width_kernel_size = kernel_size


        # padding of conv 
        self.time_pad = self.dilation * (self.time_kernel_size - 1) # 1 * (3 - 1) => 2
        self.height_pad = self.height_kernel_size // 2  # ≈ 1 
        self.width_pad = self.width_kernel_size // 2    # ≈ 1

        # (1, 1, 1, 1, 2, 0)
        self.time_causal_padding = (self.width_pad, self.width_pad, self.height_pad, self.height_pad, self.time_pad, 0)

        # print(f"in the conv class: kernel_size: {kernel_size}, stride: {stride}, padding: {padding}, dilation: {self.dilation}")
        self.conv = nn.Conv3d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=self.dilation,
                              **kwargs)
        
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)




    def forward(self, 
                x: torch.FloatTensor,
                is_init_image: bool = True,
                temporal_chunk: bool = False):
        
        pad_mode = self.pad_mode if self.time_pad < x.shape[2] else 'constant'

        if is_init_image:
            if not temporal_chunk:
                # padding (1, 1, 1, 1, 2, 0) -> (height, width, frame) 
                # (8) + (2, 0) => 10
                # (256, 256) + (1, 1, 1, 1) => (258, 258)
                # ([1, 2, 8, 256, 256])  ->  ([2, 3, 10, 258, 258])
                x = nn.functional.pad(input=x,
                                    pad=self.time_causal_padding,
                                    mode=pad_mode)
                
            else:
                print(f'work in progress... `{temporal_chunk}`')

        
        x = self.conv(x)
        return x 
    


class CausalGroupNorm(nn.GroupNorm):

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
    
        t = x.shape[2]
        x = rearrange(tensor=x,
                      pattern='b c t h w -> (b t) c h w')
        
        # pass the data in GroupNorm 
        x = super().forward(x)
        x = rearrange(tensor=x,
                      pattern='(b t) c h w -> b c t h w', t=t)
        
        return x 




if __name__ == "__main__":

    conv_layer = CausalConv3d(in_channels=3,
                              out_channels=3,
                              kernel_size=(3, 3, 3),
                              stride=1,
                              padding=0)
    # print(conv_layer)
    print(conv_layer.apply(conv_layer._init_weights))

    x = torch.randn(2, 3, 8, 256, 256)

    output = conv_layer(x)
    print(output.shape) 


        