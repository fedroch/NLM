from torch import nn
from torch.nn.functional import normalize
IN_DIM = 1024
HIDDEN1 = 1024
HIDDEN2 = 1024
OUT_DIM = 128   # дефолт; фактическая размерность приходит из чекпоинта (d_out)
class percp(nn.Module):
    def __init__(self, d_in=IN_DIM, d_out=OUT_DIM, hidden1=HIDDEN1, hidden2=HIDDEN2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden1),
            nn.GELU(),
            nn.Linear(hidden1, hidden2),
            nn.GELU(),
            nn.Linear(hidden2, d_out),
        )

    def forward(self, x):
        x = self.net(x)
        return normalize(x)
    
