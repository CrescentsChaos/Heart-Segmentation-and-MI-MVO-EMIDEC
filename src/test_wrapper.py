import torch
from monai.inferers import sliding_window_inference

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_pathology_by_disease = True
        self.disease_threshold = 0.5
        
    def forward(self, x):
        # spatial tensor
        anat = x.clone()
        path = x.clone()
        # scalar
        dp = torch.ones(x.shape[0], 1) * 0.8
        
        return {
            "anatomy_logits": anat,
            "pathology_prob": path,
            "disease_prob": dp,
            "disease_logits": dp
        }

def run_eval_patchwise(
    model, volume, patch_size=(16, 64, 64), sw_batch_size=2, overlap=0.5
):
    orig_gate = getattr(model, "gate_pathology_by_disease", False)
    if orig_gate:
        model.gate_pathology_by_disease = False

    disease_probs = []

    def predictor(patch):
        out = model(patch)
        if "disease_prob" in out:
            disease_probs.append(out["disease_prob"])
        return {k: v for k, v in out.items() if v.ndim >= 4}

    out = sliding_window_inference(
        inputs=volume,
        roi_size=patch_size,
        sw_batch_size=sw_batch_size,
        predictor=predictor,
        overlap=overlap,
        mode="gaussian",
    )

    if orig_gate:
        model.gate_pathology_by_disease = True

    if disease_probs:
        avg_prob = torch.cat(disease_probs, dim=0).mean(dim=0, keepdim=True)
        out["disease_prob"] = avg_prob
        
        if orig_gate:
            disease_gate = (avg_prob > model.disease_threshold).float()
            disease_gate = disease_gate.view(-1, 1, 1, 1, 1)
            if "pathology_prob" in out:
                out["pathology_prob"] = out["pathology_prob"] * disease_gate
            out["disease_gate"] = disease_gate

    return out

m = DummyModel()
x = torch.randn(1, 1, 16, 96, 96)
res = run_eval_patchwise(m, x)
print("Keys:", res.keys())
print("Pathology prob shape:", res["pathology_prob"].shape)
print("Disease prob shape:", res["disease_prob"].shape)
print("Disease prob value:", res["disease_prob"][0].item())
