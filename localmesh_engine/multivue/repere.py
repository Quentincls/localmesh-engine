"""Explicit coordinate conversion, before fresh shape sampling and after decode.

This never rotates existing learned latent feature channels.
"""
def structure_to_pixal(coords, grid_resolution=32):
    out = coords[:, [0,1,3,2]].clone()
    out[:,3] = grid_resolution-1-out[:,3]
    return out.contiguous()


def decoded_to_local(vertices, grid_resolution=32, decoder_grid=64):
    center = grid_resolution/(2*decoder_grid)-.5
    out = vertices[:,[0,2,1]].clone()
    out[:,1] = -out[:,1]+2*center
    return out.contiguous()
