"""
SAM 2 segmentation backend on Modal (serverless GPU).

Deploy:  modal deploy scripts/modal_sam.py
Test:    modal run scripts/modal_sam.py

After deploying, set the web endpoint URL as MODAL_SAM_URL in webapp/.env.local
and in Vercel environment variables.
"""

import modal

SAM2_GIT_SHA = "c2ec8e14a185632b0a5d8b161928ceb50197eddc"
MODEL_ID = "facebook/sam2-hiera-base-plus"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "python3-opencv")
    .pip_install(
        "torch",
        "torchvision",
        "opencv-python-headless",
        "numpy",
        "Pillow",
        "requests",
        f"git+https://github.com/facebookresearch/sam2.git@{SAM2_GIT_SHA}",
    )
)

app = modal.App("solar-labeling-sam", image=image)

hf_cache = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
CACHE_DIR = "/root/.cache/huggingface"


@app.cls(
    gpu="T4",
    container_idle_timeout=300,
    volumes={CACHE_DIR: hf_cache},
)
class SAMPredictor:
    @modal.enter()
    def load_model(self):
        import torch
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.predictor = SAM2ImagePredictor.from_pretrained(MODEL_ID)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        hf_cache.commit()  # Persist downloaded model weights

    @modal.web_endpoint(method="POST")
    def segment(self, data: dict):
        import numpy as np
        import cv2
        import torch
        from PIL import Image
        import requests
        from io import BytesIO

        image_url = data["image_url"]
        points = data["points"]  # [[x, y, label], ...] label=1 positive, 0 negative

        # Download image
        response = requests.get(image_url, timeout=30)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        image_np = np.array(img)

        # Predict
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.predictor.set_image(image_np)

            point_coords = np.array([[p[0], p[1]] for p in points])
            point_labels = np.array([p[2] for p in points])

            masks, scores, _ = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )

        # Take best mask
        best_idx = int(np.argmax(scores))
        mask = masks[best_idx]

        # Convert mask to polygon contours
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        polygons = []
        for contour in contours:
            if len(contour) < 3:
                continue
            epsilon = 0.002 * cv2.arcLength(contour, True)
            simplified = cv2.approxPolyDP(contour, epsilon, True)
            if len(simplified) >= 3:
                polygons.append(simplified.squeeze().tolist())

        return {
            "polygons": polygons,  # [[x, y], ...] in pixel coordinates
            "score": float(scores[best_idx]),
            "image_size": [int(image_np.shape[1]), int(image_np.shape[0])],
        }


@app.local_entrypoint()
def main():
    """Quick test of the SAM endpoint."""
    import json

    predictor = SAMPredictor()
    result = predictor.segment.remote(
        {
            "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/300px-PNG_transparency_demonstration_1.png",
            "points": [[150, 150, 1]],
        }
    )
    print(json.dumps(result, indent=2))
