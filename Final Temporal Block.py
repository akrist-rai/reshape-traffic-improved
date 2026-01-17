short, mid, long = build_multiscale_inputs(x)

f_s = self.temporal_short(short)
f_m = self.temporal_mid(mid)
f_l = self.temporal_long(long)

temporal_out = self.scale_attention([f_s, f_m, f_l])
