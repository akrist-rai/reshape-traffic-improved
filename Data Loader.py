def build_multiscale_inputs(x):
    # x: [B, T, N, F] where T >= 48
    short = x[:, -12:]                       # recent 1 hour
    mid   = x[:, -24::2]                     # downsample (10 min)
    long  = x[:, -48::4]                     # downsample (20 min)
    return short, mid, long
