from pathlib import Path
import json

import torch
import torchvision.models as models
import onnxruntime as ort


PROJECT_DIR = Path(__file__).resolve().parent
REGISTRY_PATH = PROJECT_DIR / "model_registry.json"


def load_registry():
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_class_mapping(mapping_path):
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    mapping = {str(k): int(v) for k, v in mapping.items()}

    if sorted(mapping.values()) != list(range(len(mapping))):
        raise ValueError("Invalid class index mapping.")

    return mapping


class SharedModel:
    def __init__(self, model_name, device="cpu"):
        registry = load_registry()

        if model_name not in registry:
            raise ValueError(f"Unknown model: {model_name}")

        self.name = model_name
        self.config = registry[model_name]

        if not self.config.get("enabled", False):
            raise ValueError(f"Model is disabled: {model_name}")

        self.backend = self.config["backend"]
        self.architecture = self.config["architecture"]
        self.num_classes = self.config["num_classes"]
        self.device = torch.device(device)

        checkpoint_path = PROJECT_DIR / self.config["checkpoint"]

        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)

        if self.backend == "pytorch":
            self._load_pytorch(checkpoint_path)

        elif self.backend == "onnxruntime":
            self._load_onnx(checkpoint_path)

        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _load_pytorch(self, checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True
        )

        if self.architecture == "mobilenet_v3_large":
            model = models.mobilenet_v3_large(weights=None)
            model.classifier[3] = torch.nn.Linear(
                model.classifier[3].in_features,
                self.num_classes
            )
        else:
            raise ValueError(
                f"Unsupported architecture: {self.architecture}"
            )

        state_dict = checkpoint.get(
            "model_state_dict",
            checkpoint.get("state_dict")
        )

        if state_dict is None:
            raise ValueError("Checkpoint has no state dictionary.")

        checkpoint_arch = checkpoint.get("architecture")

        if checkpoint_arch and checkpoint_arch != self.architecture:
            raise ValueError("Checkpoint architecture mismatch.")

        self.class_to_idx = {
            str(k): int(v)
            for k, v in checkpoint["class_to_idx"].items()
        }

        if sorted(self.class_to_idx.values()) != list(
            range(self.num_classes)
        ):
            raise ValueError("Invalid checkpoint class mapping.")

        model.load_state_dict(state_dict, strict=True)
        self.model = model.to(self.device).eval()

    def _load_onnx(self, checkpoint_path):
        if self.device.type != "cpu":
            raise ValueError("ONNX deployment currently supports CPU only.")

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1

        self.session = ort.InferenceSession(
            str(checkpoint_path),
            sess_options=options,
            providers=["CPUExecutionProvider"]
        )

        mapping_path = PROJECT_DIR / self.config["class_mapping"]
        self.class_to_idx = load_class_mapping(mapping_path)

        if len(self.class_to_idx) != self.num_classes:
            raise ValueError("ONNX class count mismatch.")

    @torch.inference_mode()
    def predict_logits(self, images):
        if self.backend == "pytorch":
            images = images.to(self.device)
            return self.model(images).cpu()

        if self.backend == "onnxruntime":
            inputs = images.detach().cpu().numpy().astype("float32")

            outputs = self.session.run(
                None,
                {self.session.get_inputs()[0].name: inputs}
            )

            return torch.from_numpy(outputs[0])

        raise RuntimeError("Unsupported inference backend.")
