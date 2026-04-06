import torch.nn as nn
from macro_place.objective import compute_proxy_cost

class PlacementGuidance(nn.Module):
    def __init__(self, benchmark, plc, wl_w=1.0, density_w=0.5, congestion_w=0.5):
        super().__init__()
        self.benchmark = benchmark
        self.plc = plc
        self.wl_w = wl_w
        self.density_w = density_w
        self.congestion_w = congestion_w

    def forward(self, positions, obv=None):
        # positions: [num_macros, 2]
        costs = compute_proxy_cost(positions, self.benchmark, self.plc)
        total_cost = (
            self.wl_w * costs['wirelength_cost'] +
            self.density_w * costs['density_cost'] +
            self.congestion_w * costs['congestion_cost']
        )
        return total_cost