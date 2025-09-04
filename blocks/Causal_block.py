import torch 
from torch import nn 
from einops import rearrange
from diffusers.models.attention_processor import Attention

from resnets.Causal_resnet import CausalResnet3d, CausalTemporalUpsample2x, CausalUpsample2x

class CausalMidBlock3d(nn.Module):

    def __init__(self,
                 in_channels: int,
                 attention_head_dim: int = 512,
                 num_groups: int = 32,
                 add_attention: bool = True,
                 dropout: float = 0.0,
                 num_layers: int = 1,
                 ):
        
        super().__init__()
        self.add_attention = add_attention

        resnets = [
            CausalResnet3d(
                in_channels=in_channels,
                out_channels=in_channels,
                num_groups=num_groups,
                dropout=dropout,
            )
        ]
        
        attentions = []
        for _ in range(num_layers):
            if self.add_attention:
                attentions.append(
                    Attention(
                    query_dim=in_channels,
                    heads=in_channels // attention_head_dim,
                    dim_head=attention_head_dim,
                    rescale_output_factor=1.0,
                    eps=1e-6,
                    norm_num_groups=num_groups,
                    spatial_norm_dim=None,
                    residual_connection=True,
                    bias=True,
                    upcast_softmax=True,
                    _from_deprecated_attn_block=True
                )
                )
            else:
                attentions.append(None)

            resnets.append(
                CausalResnet3d(
                in_channels=in_channels,
                out_channels=in_channels,
                num_groups=num_groups,
                dropout=dropout,
            )
            )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        
        


    def forward(self, 
                hidden_size: torch.FloatTensor,
                is_init_image=True,
                temporal_chunk=False):
        
        hidden_size = self.resnets[0](hidden_size, is_init_image, temporal_chunk)

        # [batch_size, channels, time, height, width]
        t = hidden_size.shape[2]

        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                hidden_size = rearrange(hidden_size, 'b c t h w -> b t c h w')
                hidden_size = rearrange(hidden_size, 'b t c h w -> (b t) c h w')
                hidden_size = attn(hidden_size)
                hidden_size = rearrange(hidden_size, '(b t) c h w -> b t c h w', t=t)
                hidden_size = rearrange(hidden_size, 'b t c h w -> b c t h w')

            hidden_size = resnet(hidden_size, is_init_image, temporal_chunk)
        
        return hidden_size

        
        return hidden_size
    

class CausalUpBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_layers: int = 3,
                 num_groups: int = 32,
                 dropout: float = 0.0,
                 add_spatial_upsample: bool = True,
                 add_temporal_upsample: bool = False):
        

        super().__init__()


        self.resnets = nn.ModuleList([])
        for i in range(num_layers):
            input_channels = in_channels if i==0 else out_channels

            self.resnets.append(
                CausalResnet3d(in_channels=input_channels,
                                    out_channels=out_channels,
                                    num_groups=num_groups,
                                    dropout=dropout)
            
            )

        if add_spatial_upsample:
            self.upsamplers = nn.ModuleList([
                CausalUpsample2x(in_channels=out_channels,
                                 out_channels=out_channels)
            ])

        else:
            self.upsamplers = None

        if add_temporal_upsample:
            self.temporal_upsamplers = nn.ModuleList([
                CausalTemporalUpsample2x(in_channels=out_channels,
                                         out_channels=out_channels)
            ])
        else:
            self.temporal_upsamplers = None





    
    def forward(self, 
                hidden_state: torch.FloatTensor,
                is_init_image=True,
                temporal_chunk=False) -> torch.FloatTensor:
        
        for resnet in self.resnets:
            hidden_state = resnet(hidden_state,
                                  is_init_image,
                                  temporal_chunk)
            
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_state = upsampler(hidden_state,
                                         is_init_image,
                                         temporal_chunk)
                
        if self.temporal_upsamplers is not None:
            for temporal_upsampler in self.temporal_upsamplers:
                hidden_state = temporal_upsampler(hidden_state,
                                                  is_init_image,
                                                  temporal_chunk)
        
        return hidden_state
        