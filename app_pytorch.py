"""Gradio demo for seven-class disease and categorical Acne/Eczema severity."""

from pathlib import Path
import os

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from infer import (
    IMAGE_SIZE,
    STAGE1_CLASSES,
    SkinDiagnosticPipeline,
)


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, target_class):
        self.model.eval()
        self.model.zero_grad()
        logits = self.model(input_tensor)
        score = logits[0, target_class]
        score.backward()
        weights = self.gradients.detach().mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations.detach()).sum(dim=1)).squeeze().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        else:
            cam = np.zeros_like(cam)
        return cam


def overlay_cam(image, cam, alpha=0.45):
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image_array = np.asarray(image).astype(np.float32) / 255.0
    cam_image = Image.fromarray((cam * 255).astype(np.uint8)).resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR
    )
    cam_array = np.asarray(cam_image).astype(np.float32) / 255.0
    colored_cam = plt.get_cmap("jet")(cam_array)[:, :, :3]
    return np.clip((1.0 - alpha) * image_array + alpha * colored_cam, 0.0, 1.0)


BASE_DIR = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIPELINE = SkinDiagnosticPipeline(
    stage1_path=str(BASE_DIR / "runs" / "resnet50_v3" / "best_model.pt"),
    acne_severity_path=str(BASE_DIR / "runs" / "severity_acne" / "best_model.pt"),
    eczema_severity_path=str(BASE_DIR / "runs" / "severity_eczema" / "best_model.pt"),
    device=str(DEVICE),
)
GRAD_CAM = GradCAM(PIPELINE.stage1_model, PIPELINE.stage1_model.layer4[-1].conv3)


def predict_image(image):
    if image is None:
        return "Upload an image to begin.", {}, None

    result = predict_image_fallback(image)

    disease_index = STAGE1_CLASSES.index(result["disease"])
    tensor = PIPELINE.transforms(image).unsqueeze(0).to(PIPELINE.device)
    cam = GRAD_CAM.generate(tensor, disease_index)
    overlay = overlay_cam(image, cam)

    display_result = dict(result)
    if result["disease_confidence"] < 0.6:
        display_result["display_disease"] = "Uncertain - low confidence prediction"
        display_result["display_severity"] = "Not reported"
        display_result["severity"] = "N/A"
    else:
        display_result["display_disease"] = result["disease"]
        display_result["display_severity"] = result["severity"]

    summary = (
        f"### {display_result['display_disease']}\n"
        f"Disease confidence: {result['disease_confidence'] * 100:.2f}%\n\n"
        f"Severity: {display_result['display_severity']}"
    )
    return summary, display_result, overlay


def predict_image_fallback(image):
    return PIPELINE.predict_image(image)


with gr.Blocks(title="Skin Disease Assessment") as demo:
    gr.Markdown("# Skin Disease Detection and Severity Assessment")
    with gr.Row():
        input_image = gr.Image(type="pil", label="Upload image")
        cam_output = gr.Image(label="Grad-CAM overlay")
    result_output = gr.Markdown()
    details_output = gr.JSON(label="Prediction details")
    analyze_button = gr.Button("Analyze image")
    analyze_button.click(predict_image, inputs=input_image, outputs=[result_output, details_output, cam_output])


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        show_error=True,
    )