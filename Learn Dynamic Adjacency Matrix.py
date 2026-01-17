#Instead of fixed A, learn A_t.
  
class DynamicGraphLearner(nn.Module):
    def __init__(self, dim, N):
        super().__init__()
        self.node_emb = nn.Parameter(torch.randn(N, dim))
        self.proj = nn.Linear(dim, dim)

    def forward(self, h):
        # h: [B, N, F]
        q = self.proj(h)
        k = self.node_emb.unsqueeze(0)
        A = torch.softmax(torch.matmul(q, k.transpose(-1, -2)), dim=-1)
        return A
