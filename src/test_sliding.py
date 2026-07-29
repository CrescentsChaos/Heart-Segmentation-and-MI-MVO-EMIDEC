import torch
from monai.inferers import sliding_window_inference

class DummyModel(torch.nn.Module):
    def forward(self, x):
        return {"out1": x, "out2": x * 2}

m = DummyModel()
x = torch.randn(1, 1, 16, 96, 96)
try:
    res = sliding_window_inference(x, (16, 64, 64), 1, m, overlap=0.5)
    print("Success:", res.keys())
except Exception as e:
    print("Failed:", type(e), e)
