#Let the model learn how much to trust each scale.

class TemporalScaleAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, feats):
        # feats: list of [B, N, F]
        scores = [self.attn(f) for f in feats]
        weights = torch.softmax(torch.stack(scores), dim=0)
        fused = sum(w * f for w, f in zip(weights, feats))
        return fused
