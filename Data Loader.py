#this is the improvenment snipt (full code (not including this) in reshaping traffic  )
def build_multiscale_inputs(x):
    # x: [B, T, N, F] where T >= 48
    short = x[:, -12:]                       # recent 1 hour
    mid   = x[:, -24::2]                     # downsample (10 min)
    long  = x[:, -48::4]                     # downsample (20 min)
    return short, mid, long



# a time step is 5 min so 12 previous time steps = 1 hour 
# when we take the data of past of 4 hours that is the last loop we take alternate 4th value so that window match
# the key is that now the model knows that short term momory and long term too
