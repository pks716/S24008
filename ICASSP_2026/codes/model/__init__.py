from .simpleunet import SimpleUNet, SimpleDiscriminator
from .attentiondenseunet import AttentionDenseUNet
from .umamba_bi import AttentionUMambaUNet
from .umamba_bi_tmi import AttentionUMambaUNet as AttentionUMambaUNetImprovedTMI

def build_network(network_opt):
    net_type = network_opt['type']
    params = network_opt['params']
    if net_type == 'simpleunet':
        return SimpleUNet(**params)
    elif net_type == 'simpledisc':
        return SimpleDiscriminator(**params)
    elif net_type == 'attentiondenseunet':
        return AttentionDenseUNet(**params)
    elif net_type == 'umamba_bi':
        return AttentionUMambaUNet(**params)
    elif net_type == 'umamba_bi_tmi':
        return AttentionUMambaUNetImprovedTMI(**params)
    else:
        raise ValueError(f'Unknown network type: {net_type}')