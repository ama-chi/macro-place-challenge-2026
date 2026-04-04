import torch
from torch.utils.data import DataLoader

from submissions.stochastic_interpolant.si import StochasticInterpolant

class SiTrainer:
    def __init__(
        self,
        model: StochasticInterpolant,
        train_loader: DataLoader,
        epochs: int,
        lr: float,
        grad_clip: float = 1.0,
        enable_amp: bool = True,
        lr_s: float | None = None,
        lr_b: float | None = None
    ):
        self.model = model
        self.device = self.model.device
        self.train_loader = train_loader
        self.epochs = epochs
        self.grad_clip = grad_clip

        self.opt_v = torch.optim.AdamW(
            self.model.v_model.parameters(), lr = lr
        )

        self.opt_s = torch.optim.AdamW(
            self.model.s_model.parameters(), lr = lr * 0.5 if lr_s is None else lr_s
        )

        if self.model.train_b:
            self.opt_b = torch.optim.AdamW(
                self.model.b_model.parameters(), lr = lr * 0.5 if lr_b is None else lr_b
            )
        
        self.enable_amp = enable_amp
        self.scaler = torch.amp.GradScaler(device_type=self.device, enabled=self.enable_amp)

    def step(self, x0: torch.Tensor, x1: torch.Tensor, obv: torch.Tensor):
        self.opt_v.zero_grad(set_to_none=True)
        self.opt_s.zero_grad(set_to_none=True)
        if self.model.train_b:
            self.opt_b.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=self.device, enabled=self.enable_amp):
            loss = self.model.loss(x0, x1, obv)

        self.scaler.scale(loss).backward()

        self.scaler.unscale_(self.opt_v)
        self.scaler.unscale_(self.opt_s)
        if self.model.train_b:
            self.scaler.unscale_(self.opt_b)

        torch.nn.utils.clip_grad_norm_(
            self.model.v_model.parameters(), self.grad_clip
        )
        torch.nn.utils.clip_grad_norm_(
            self.model.s_model.parameters(), self.grad_clip
        )
        if self.model.train_b:
            torch.nn.utils.clip_grad_norm_(
                self.model.b_model.parameters(), self.grad_clip
        )
             
        self.scaler.step(self.opt_v)
        self.scaler.step(self.opt_s)

        if self.model.train_b:
             self.scaler.step(self.opt_b)
            
        self.scaler.update()

        return loss.item()

    def train(self):
        self.model.v_model.train()
        self.model.s_model.train()
        if self.model.train_b:
            self.model.b_model.train()

        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch in self.train_loader:
                # batch = (x0, x1, obv)
                x0, x1, obv = batch

                loss = self.step(x0, x1, obv)
                total_loss += loss

            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch [{epoch+1}/{self.epochs}] Loss: {avg_loss:.6f}")