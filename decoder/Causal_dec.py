import torch
from torch import nn 
from typing import Union, Tuple

from conv import CausalConv3d, CausalGroupNorm
from blocks.Causal_block import CausalMidBlock3d, CausalUpBlock

## one things to understand
# the encoder input is [batch_size, in_channels, frame, height, width] =>  [2, 3, 8, 256, 256] => video 
# so the output to get [batch_size, 2*out_channels, frame, height, width => [2, 6, 1, 32, 32] => latent image 



class CausalDecoder(nn.Module):

    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 kernel_size: Union[int, Tuple[int, ...]] = 3,
                 stride: Union[int, Tuple[int, ...]] = 1,
                 padding: Union[int, Tuple[int, ...]] = 1,
                 decoder_types: Union[str, Tuple[str, ...]] = ("CausalUpBlock3d",),
                 block_out_channels: Tuple[int, ...] = (128,),
                 num_layers: int = 2,
                 num_groups: int = 32,
                 dropout: float = 0.0,
                 add_spatial_upsample: Tuple[bool, ...] = (True,),
                 add_temporal_upsample: Tuple[bool, ...] = (False,)
                 ):
        

        super().__init__()

        self.conv_in = CausalConv3d(in_channels=in_channels,
                                    out_channels=block_out_channels[-1])
        

        
        self.mid_block = CausalMidBlock3d(
            in_channels=block_out_channels[-1],
            attention_head_dim=block_out_channels[-1], # 512 
            num_groups=num_groups,
            add_attention=True,
            dropout=dropout,
            
        )

        # up blocks 
        self.up_blocks = nn.ModuleList([])
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channels = reversed_block_out_channels[0]
        for i, decoder_type in enumerate(decoder_types):
            # print(f"what is the index: {i} and decoder_type: {decoder_types}")
            input_channels = output_channels
            output_channels = reversed_block_out_channels[i]

            # [512] -> [512]
            # [512] -> [512]
            # [512] -> [256]
            # [256] -> [128]
            
            up_block = CausalUpBlock(in_channels=input_channels,
                                     out_channels=output_channels,
                                     num_layers=num_layers[i],
                                     num_groups=num_groups,
                                     dropout=dropout,
                                     add_spatial_upsample=add_spatial_upsample[i],
                                     add_temporal_upsample=add_temporal_upsample[i])

            self.up_blocks.append(up_block)

        # output 
        self.conv_norm_out = CausalGroupNorm(num_groups=num_groups,
                                             num_channels=block_out_channels[0],
                                             eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = CausalConv3d(in_channels=block_out_channels[0],
                                     out_channels=out_channels,
                                     kernel_size=3,
                                     stride=1)
        
        self.gradient_checkpointing = False


    def forward(self, 
                sample: torch.FloatTensor,
                is_init_image=True,
                temporal_chunk=False):
        

        sample = self.conv_in(sample, 
                              is_init_image,
                              temporal_chunk)
        
        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
        
        # middle block 
        sample = self.mid_block(sample,
                                is_init_image,
                                temporal_chunk)
        
        # up block 
        for up_block in self.up_blocks:
            sample = up_block(sample,
                              is_init_image,
                              temporal_chunk)
            
        # post-process 
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample, is_init_image, temporal_chunk)
        

        return sample 
    


if __name__ == "__main__":

    # encoder input to output 
    # ([2, 3, 8, 256, 256]) -> ([2, 3*2, 1, 32, 32]) if double_z = True else ([2, 3, 1, 32, 32])
    causal_decoder = CausalDecoder(
        in_channels=4,
        out_channels=3,
        kernel_size=3,
        stride=1,
        padding=1,
        decoder_types=("CausalUpBlock3d",
                       "CausalUpBlock3d",
                       "CausalUpBlock3d",
                       "CausalUpBlock3d",),
        block_out_channels=(128, 256, 512, 512),
        num_layers=(3, 3, 3, 3),
        num_groups=2,
        add_spatial_upsample=(True, True, True, False),
        add_temporal_upsample=(True, True, True, False)
    )

    print(causal_decoder)

    z = torch.randn(2, 4, 1, 32, 32)

    output = causal_decoder(z)
    print(output.shape)
    