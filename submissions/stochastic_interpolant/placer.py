from submissions.stochastic_interpolant.si import StochasticInterpolant
from macro_place.benchmark import Benchmark
import torch

class StochasticInterpolantPlacer:
    def __init__(self,
                 model: StochasticInterpolant):
        self.model = model

    def load_model(self, fp: str) -> None:
        self.model.load(fp)

    def place(self, benchmark: Benchmark):

        placement = benchmark.macro_positions.clone()
        movable = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
        movable_indices = torch.where(movable)[0].tolist()

        sizes = benchmark.macro_sizes
        canvas_w = benchmark.canvas_width
        canvas_h = benchmark.canvas_height

        


        



