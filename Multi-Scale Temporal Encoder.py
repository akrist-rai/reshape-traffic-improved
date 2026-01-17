class TemporalEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mamba = Mamba(d_model=dim)

    def forward(self, x):
        # x: [B, T, N, F]
        B, T, N, F = x.shape
        x = x.view(B*N, T, F)
        out = self.mamba(x)
        return out[:, -1].view(B, N, F)
