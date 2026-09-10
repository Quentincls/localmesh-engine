"""NAF neighborhood attention only at requested output pixels.

Matches NATTEN noncausal stride-1, odd windows, integer upsampling. Shifted
boundary windows are retained; there is no zero-padding approximation.
"""
import torch
import torch.nn.functional as F


def attention_at_pixels(q_map, lr_features, pixels, heads=4, kernel=9, chunk=128):
    if q_map.shape[0] != 1 or lr_features.shape[0] != 1:
        raise ValueError('Process one view at a time')
    _, cq, height, width = q_map.shape
    _, cv, lh, lw = lr_features.shape
    if height % lh or width % lw or min(lh,lw) < kernel or kernel % 2 != 1:
        raise ValueError('Expected integer upsampling and a supported odd NATTEN window')
    if cq % heads or cv % heads:
        raise ValueError('Channels must divide attention heads')
    if pixels.numel() and (bool((pixels < 0).any()) or bool((pixels[:,0]>=height).any()) or bool((pixels[:,1]>=width).any())):
        raise ValueError('Output pixel outside the feature map')
    key = F.adaptive_avg_pool2d(q_map, (lh,lw))
    q = q_map[0].permute(1,2,0).reshape(height*width,heads,cq//heads)
    k = key[0].permute(1,2,0).reshape(lh*lw,heads,cq//heads)
    v = lr_features.to(q_map.dtype)[0].permute(1,2,0).reshape(lh*lw,heads,cv//heads)
    result = q_map.new_empty((len(pixels),heads,cv//heads))
    offsets = torch.arange(kernel,device=q.device)
    dh, dw = height//lh, width//lw
    for begin in range(0,len(pixels),chunk):
        p = pixels[begin:begin+chunk].long()
        sy = (p[:,0]//dh-kernel//2).clamp(0,lh-kernel)
        sx = (p[:,1]//dw-kernel//2).clamp(0,lw-kernel)
        neighbors = ((sy[:,None,None]+offsets[None,:,None])*lw + sx[:,None,None]+offsets[None,None,:]).flatten(1)
        queries = q[p[:,0]*width+p[:,1]]
        scores = torch.einsum('nhd,njhd->nhj',queries,k[neighbors]) * ((cq//heads)**-.5)
        weights = scores.softmax(-1)
        for channel in range(0,cv//heads,64):
            values = v[:,:,channel:channel+64][neighbors]
            result[begin:begin+len(p),:,channel:channel+64] = torch.einsum('nhj,njhd->nhd',weights,values)
    return result.reshape(len(pixels),cv)


def sample_naf_features(q_map, lr_features, queries_ndc, **kwargs):
    """Bilinear border sampling of the implicit NAF HR map; [N,2] -> [N,C]."""
    height, width = q_map.shape[-2:]
    x = ((queries_ndc[:,0]+1)*width/2-.5).clamp(0,width-1)
    y = ((queries_ndc[:,1]+1)*height/2-.5).clamp(0,height-1)
    x0, y0 = x.floor().long(), y.floor().long()
    x1, y1 = (x0+1).clamp(max=width-1), (y0+1).clamp(max=height-1)
    corners = torch.stack((torch.stack((y0,x0),-1),torch.stack((y0,x1),-1),
                           torch.stack((y1,x0),-1),torch.stack((y1,x1),-1)),dim=1)
    unique, inverse = torch.unique(corners.reshape(-1,2),dim=0,return_inverse=True)
    values = attention_at_pixels(q_map,lr_features,unique,**kwargs)[inverse].reshape(len(x),4,-1)
    wx, wy = x-x0, y-y0
    weights = torch.stack(((1-wx)*(1-wy),wx*(1-wy),(1-wx)*wy,wx*wy),dim=1)
    return (values*weights[:,:,None]).sum(1)
