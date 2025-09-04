import torch 
from torch import nn 
from einops import rearrange

from conv import CausalConv3d, CausalGroupNorm

class CausalResnet3d(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_groups: int = 32,
                 dropout: float = 0.0,
                 use_in_shortcut: bool = None
                 ):
        super().__init__()

        
        # [128]
        self.norm1 = CausalGroupNorm(num_groups=num_groups,
                                     num_channels=in_channels,
                                     eps=1e-6)
        
        self.conv1 = CausalConv3d(in_channels=in_channels,
                                  out_channels=out_channels)
        

        self.norm2 = CausalGroupNorm(num_groups=num_groups,
                                     num_channels=out_channels,
                                     eps=1e-6)
        self.dropout = torch.nn.Dropout(dropout)
        
        conv_2d_out_channels = out_channels
        self.conv2 = CausalConv3d(in_channels=out_channels,
                                  out_channels=out_channels)
        
        self.activation_fn = nn.SiLU()

        # this is true when in_channels != conv_2d_out_channels 
        # [128] != [256] => True 
        # [256] != [512] => True 
        if use_in_shortcut is None:
            self.use_in_shortcut = in_channels != conv_2d_out_channels

        else:
            self.use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = CausalConv3d(in_channels=in_channels,
                                              out_channels=conv_2d_out_channels,
                                              kernel_size=1,
                                              stride=1,
                                              bias=True
                                              )
            
        else:
            self.use_in_shortcut = False


        
    def forward(self, 
                x,
                is_init_image=True,
                temporal_chunk=False):
        

        input_tensor = x 
        x = self.norm1(x)
        x = self.activation_fn(x)
        x = self.conv1(x, is_init_image, temporal_chunk)

        x = self.norm2(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.conv2(x, is_init_image, temporal_chunk)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor, is_init_image, temporal_chunk)

        output_tensor = (input_tensor + x) / 1.0

        return output_tensor


class CausalUpsample2x(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 ):
        
        super().__init__()
        self.in_channels = in_channels

        # [512] -> [4*256]
        self.conv = CausalConv3d(in_channels=in_channels,
                                 out_channels=out_channels * 4,
                                 kernel_size=3,
                                 stride=1,
                                 bias=True)
        
    
    def forward(self,
                hidden_state: torch.FloatTensor,
                is_init_image=True,
                temporal_chunk=False) -> torch.FloatTensor:
        
        assert hidden_state.shape[1] == self.in_channels, 'make sure `video channels` is equal to `in_channels`!'
        hidden_state = self.conv(hidden_state,
                                 is_init_image,
                                 temporal_chunk)
        
        # [2, (256*2*2), 8, 256, 256] -> [2, 256, 8, (2*256), (2*256)]
        hidden_state = rearrange(hidden_state,
                                 'b (c p1 p2) t h w -> b c t (h p1) (w p2)',
                                 p1=2, p2=2)
        
        return hidden_state
    

class CausalTemporalUpsample2x(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        
        self.conv = CausalConv3d(in_channels=in_channels,
                            out_channels=out_channels * 2,
                            kernel_size=3,
                            stride=1,
                            bias=True)
        
    def forward(self, 
                hidden_states: torch.FloatTensor,
                is_init_image=True,
                temporal_chunk=False) -> torch.FloatTensor:
        
        assert hidden_states.shape[1] == self.in_channels, 'make sure `video_channels` is equal to `in_channels`!'

        t = hidden_states.shape[2]
        hidden_states = self.conv(hidden_states,
                                  is_init_image,
                                  temporal_chunk)
        
        # [2, (256*2), 8, 256, 256] -> [2, 512, (8*2), 256, 256]
        hidden_states = rearrange(hidden_states,
                                  'b (c p) t h w -> b c (t p) h w', t=t, p=2)
        

        if is_init_image:
            # [2, (256*2), 8, 256, 256] -> [2, 512, (8*2-1), 256, 256]
            hidden_states = hidden_states[:, :, 1:]

        return hidden_states

