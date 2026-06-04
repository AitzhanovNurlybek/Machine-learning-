"""
SIS4 – Image Classification using Pretrained Models
Deep Learning Lab – Complete Solution
Author: Aitzhanov Nurlybek
"""

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import requests
import time
import os
import json
import matplotlib
matplotlib.use("Agg")          # headless backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# GLOBAL SETUP


os.makedirs("images",  exist_ok=True)
os.makedirs("results", exist_ok=True)

# Standard ImageNet preprocessing
PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229,  0.224, 0.225]),
])

def load_imagenet_labels() -> list[str]:
    """Download class labels from PyTorch Hub."""
    url = ("https://raw.githubusercontent.com/pytorch/hub"
           "/master/imagenet_classes.txt")
    return requests.get(url, timeout=15).text.strip().split("\n")

LABELS = load_imagenet_labels()



# IMAGE DOWNLOAD


IMAGE_URLS = {
    # Full-resolution originals – no thumbnail API needed
    "dog":      "https://upload.wikimedia.org/wikipedia/commons/2/26/YellowLabradorLooking_new.jpg",
    "cat":      "https://upload.wikimedia.org/wikipedia/commons/4/4d/Cat_November_2010-1a.jpg",
    "car":      "https://upload.wikimedia.org/wikipedia/commons/1/1b/2019_Honda_Civic_sedan.jpg",
    "airplane": "https://upload.wikimedia.org/wikipedia/commons/8/8b/Airbus_A380_blue_sky.jpg",
    "banana":   "https://upload.wikimedia.org/wikipedia/commons/8/8a/Banana-Whole-and-Split.jpg",
}

def download_images(save_dir: str = "images") -> dict[str, str]:
    """Download test images; return mapping of name→path."""
    paths = {}
    headers = {"User-Agent": "Mozilla/5.0"}
    for name, url in IMAGE_URLS.items():
        path = os.path.join(save_dir, f"{name}.jpg")
        if os.path.exists(path):
            paths[name] = path
            print(f"  [cached] {name}.jpg")
            continue
        try:
            r = requests.get(url, headers=headers, timeout=20)
            r.raise_for_status()
            with open(path, "wb") as f:
                f.write(r.content)
            print(f"  [ok] {name}.jpg  ({len(r.content)//1024} KB)")
            paths[name] = path
        except Exception as e:
            print(f"  [fail] {name}: {e}")
    return paths


# TASK 1 – BASIC CLASSIFICATION


def task1_basic_classification(image_paths: dict[str, str]) -> dict[str, str]:
    """
    Load ResNet-50 (pretrained) and return the top-1 predicted class
    for each image.
    """
    print("\n" + "="*60)
    print("TASK 1 – Basic Image Classification (ResNet-50)")
    print("="*60)

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.eval()

    results: dict[str, str] = {}
    for name, path in image_paths.items():
        img = Image.open(path).convert("RGB")
        tensor = PREPROCESS(img).unsqueeze(0)

        with torch.no_grad():
            output = model(tensor)

        _, idx = torch.max(output, 1)
        predicted_class = LABELS[idx.item()]
        results[name] = predicted_class
        print(f"  {name:12s} → {predicted_class}")

    # Save table
    _print_table(
        headers=["Image", "Predicted Class"],
        rows=[[n, c] for n, c in results.items()],
        title="Task 1 – Top-1 Predictions",
    )
    return results



# TASK 2 – TOP-K PREDICTIONS


def classify_image(image_path: str, model: torch.nn.Module,
                   top_k: int = 5) -> list[dict]:
    """Return top-k predictions with probabilities."""
    img = PREPROCESS(Image.open(image_path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        output = model(img)
    probs = torch.nn.functional.softmax(output[0], dim=0)
    top_probs, top_idxs = torch.topk(probs, top_k)
    return [
        {"class": LABELS[idx.item()], "probability": prob.item() * 100}
        for prob, idx in zip(top_probs, top_idxs)
    ]


def task2_topk_predictions(image_paths: dict[str, str]) -> dict:
    """
    Display the original image + top-5 predictions for each test image.
    Saves a matplotlib figure to results/task2_topk.png.
    """
    print("\n" + "="*60)
    print("TASK 2 – Top-5 Predictions with Confidence Scores")
    print("="*60)

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.eval()

    all_preds = {}
    n = len(image_paths)
    fig = plt.figure(figsize=(14, 4 * n))
    gs = gridspec.GridSpec(n, 2, figure=fig, wspace=0.05, hspace=0.5)

    for row, (name, path) in enumerate(image_paths.items()):
        preds = classify_image(path, model, top_k=5)
        all_preds[name] = preds

        # Left: image
        ax_img = fig.add_subplot(gs[row, 0])
        ax_img.imshow(Image.open(path).convert("RGB"))
        ax_img.set_title(f"Input: {name}", fontsize=11, fontweight="bold")
        ax_img.axis("off")

        # Right: horizontal bar chart
        ax_bar = fig.add_subplot(gs[row, 1])
        classes = [p["class"].replace("_", " ") for p in reversed(preds)]
        probs   = [p["probability"] for p in reversed(preds)]
        colors  = ["#2196F3" if i == len(preds)-1 else "#90CAF9"
                   for i in range(len(preds))]
        bars = ax_bar.barh(classes, probs, color=colors, edgecolor="white")
        for bar, pct in zip(bars, probs):
            ax_bar.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                        f"{pct:.1f}%", va="center", fontsize=9)
        ax_bar.set_xlim(0, max(probs) * 1.2)
        ax_bar.set_xlabel("Confidence (%)")
        ax_bar.set_title(f"Top-5 Predictions – {name}", fontsize=10)
        ax_bar.spines[["top","right"]].set_visible(False)

        # Console output
        print(f"\n  [{name.upper()}]")
        for i, p in enumerate(preds, 1):
            print(f"    {i}. {p['class']:35s} {p['probability']:6.2f}%")

    fig.savefig("results/task2_topk.png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    print("\n  Figure saved → results/task2_topk.png")
    return all_preds



# TASK 3 – RESNET MODEL COMPARISON


def task3_model_comparison(image_paths: dict[str, str]) -> dict:
    """
    Compare ResNet-18, ResNet-50, and ResNet-101 on:
      - Number of parameters
      - Average inference time (CPU)
      - Top-1 accuracy on the 5 test images
    """
    print("\n" + "="*60)
    print("TASK 3 – ResNet Model Comparison")
    print("="*60)

    configs = {
        "ResNet-18":  (models.resnet18,  models.ResNet18_Weights.IMAGENET1K_V1),
        "ResNet-50":  (models.resnet50,  models.ResNet50_Weights.IMAGENET1K_V1),
        "ResNet-101": (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V1),
    }

    comparison: dict[str, dict] = {}

    for model_name, (fn, weights) in configs.items():
        model = fn(weights=weights)
        model.eval()

        # Parameter count
        n_params = sum(p.numel() for p in model.parameters())

        # Warm-up run
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            _ = model(dummy)

        # Inference timing
        times, preds = [], {}
        for img_name, path in image_paths.items():
            tensor = PREPROCESS(Image.open(path).convert("RGB")).unsqueeze(0)
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model(tensor)
            times.append((time.perf_counter() - t0) * 1000)
            _, idx = torch.max(out, 1)
            preds[img_name] = LABELS[idx.item()]

        avg_ms = sum(times) / len(times)
        comparison[model_name] = {
            "params_M": round(n_params / 1e6, 1),
            "avg_inference_ms": round(avg_ms, 1),
            "per_image_ms": [round(t, 1) for t in times],
            "top1_predictions": preds,
        }

        print(f"\n  {model_name}")
        print(f"    Parameters:     {n_params/1e6:.1f} M")
        print(f"    Avg inference:  {avg_ms:.1f} ms")
        for img_name, pred in preds.items():
            print(f"    {img_name:12s}→ {pred}")

    # Comparison table
    _print_table(
        headers=["Model", "Params (M)", "Avg Inference (ms)", "ImageNet Top-1 Acc*"],
        rows=[
            ["ResNet-18",  "11.7",  f"{comparison['ResNet-18']['avg_inference_ms']}",  "69.8%"],
            ["ResNet-50",  "25.6",  f"{comparison['ResNet-50']['avg_inference_ms']}",  "76.1%"],
            ["ResNet-101", "44.5",  f"{comparison['ResNet-101']['avg_inference_ms']}", "77.4%"],
        ],
        title="Task 3 – Model Comparison (* published benchmark, ImageNet val set)",
    )

    _plot_comparison(comparison)
    return comparison


def _plot_comparison(comp: dict):
    """Bar charts comparing model size vs. speed."""
    models_list = list(comp.keys())
    params   = [comp[m]["params_M"] for m in models_list]
    inf_time = [comp[m]["avg_inference_ms"] for m in models_list]
    accuracy = [69.8, 76.1, 77.4]  # published ImageNet Top-1

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    colors = ["#42A5F5", "#1E88E5", "#1565C0"]

    for ax, data, ylabel, title in zip(
        axes,
        [params, inf_time, accuracy],
        ["Parameters (M)", "Avg Inference (ms, CPU)", "Top-1 Accuracy (%)"],
        ["Model Size", "Inference Speed", "ImageNet Accuracy (published)"]
    ):
        bars = ax.bar(models_list, data, color=colors, edgecolor="white", width=0.5)
        for bar, val in zip(bars, data):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(data)*0.02,
                    f"{val}", ha="center", va="bottom", fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.spines[["top","right"]].set_visible(False)
        ax.set_ylim(0, max(data) * 1.2)

    fig.suptitle("ResNet Variant Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig("results/task3_comparison.png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    print("  Figure saved → results/task3_comparison.png")



# TASK 4 – ERROR ANALYSIS


def task4_error_analysis(image_paths: dict[str, str]) -> list[dict]:
    """
    Identify 2 images where ResNet-50 makes incorrect predictions.
    Returns a list of error analysis dicts.
    """
    print("\n" + "="*60)
    print("TASK 4 – Error Analysis")
    print("="*60)

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.eval()

    errors = []
    for name, path in image_paths.items():
        preds = classify_image(path, model, top_k=5)
        top_class = preds[0]["class"].lower().replace("_", " ")
        expected  = name.lower()

        # Check if the expected category appears in any top-5 prediction
        found_in_top5 = any(expected in p["class"].lower().replace("_", " ")
                            for p in preds)

        if not found_in_top5 and len(errors) < 2:
            errors.append({
                "image": name,
                "expected": expected,
                "top1_predicted": preds[0]["class"],
                "top1_confidence": preds[0]["probability"],
                "top5": preds,
            })
            print(f"\n  [ERROR] {name}")
            print(f"    Expected:  {expected}")
            print(f"    Predicted: {preds[0]['class']}  ({preds[0]['probability']:.1f}%)")
            print(f"    Top-5:")
            for i, p in enumerate(preds, 1):
                print(f"      {i}. {p['class']:35s} {p['probability']:.2f}%")

    if not errors:
        print("  No errors found in top-5 – try images with unusual angles or "
              "ambiguous content for this task.")
    return errors


# EXTRA TASK – BATCH PROCESSING


def extra_batch_processing(image_paths: dict[str, str],
                           n_total: int = 10) -> dict:
    """
    Process n_total images individually vs. in a single batch.
    Returns timing comparison dict.
    """
    print("\n" + "="*60)
    print("EXTRA TASK – Batch Processing vs. Individual Inference")
    print("="*60)

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.eval()

    # Repeat the 5 images to get n_total
    all_paths = (list(image_paths.values()) * ((n_total // len(image_paths)) + 1))[:n_total]
    print(f"  Processing {n_total} images …")

    #  Individual inference 
    ind_times = []
    for path in all_paths:
        tensor = PREPROCESS(Image.open(path).convert("RGB")).unsqueeze(0)
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(tensor)
        ind_times.append((time.perf_counter() - t0) * 1000)

    total_ind = sum(ind_times)
    avg_ind   = total_ind / len(ind_times)

    #  Batch inference 
    tensors = [PREPROCESS(Image.open(p).convert("RGB")) for p in all_paths]
    batch   = torch.stack(tensors)           # (N, 3, 224, 224)

    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(batch)
    total_batch = (time.perf_counter() - t0) * 1000
    avg_batch   = total_batch / len(all_paths)

    speedup = total_ind / total_batch

    result = {
        "n_images":              n_total,
        "individual_total_ms":   round(total_ind,   1),
        "individual_per_img_ms": round(avg_ind,     1),
        "batch_total_ms":        round(total_batch, 1),
        "batch_per_img_ms":      round(avg_batch,   1),
        "speedup":               round(speedup,     2),
    }

    print(f"\n  Individual – total: {total_ind:.1f} ms  |  per image: {avg_ind:.1f} ms")
    print(f"  Batch      – total: {total_batch:.1f} ms  |  per image: {avg_batch:.1f} ms")
    print(f"  Speedup:   {speedup:.2f}x")

    _plot_batch(ind_times, total_batch, speedup, n_total)
    return result


def _plot_batch(ind_times: list, total_batch: float,
                speedup: float, n: int):
    """Visualise individual vs. batch timing."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Per-image times
    ax1.bar(range(1, n+1), ind_times, color="#42A5F5", label="Individual")
    ax1.axhline(total_batch/n, color="#E53935", linestyle="--", linewidth=2,
                label=f"Batch avg ({total_batch/n:.1f} ms)")
    ax1.set_xlabel("Image index")
    ax1.set_ylabel("Inference time (ms)")
    ax1.set_title("Per-image Inference Time")
    ax1.legend()
    ax1.spines[["top","right"]].set_visible(False)

    # Speedup gauge
    categories = ["Individual\n(total)", "Batch\n(total)"]
    totals = [sum(ind_times), total_batch]
    bars = ax2.bar(categories, totals, color=["#42A5F5", "#43A047"],
                   width=0.4, edgecolor="white")
    for bar, val in zip(bars, totals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 f"{val:.0f} ms", ha="center", fontweight="bold")
    ax2.set_ylabel("Total time (ms)")
    ax2.set_title(f"Total Processing Time  (speedup: {speedup:.2f}×)")
    ax2.spines[["top","right"]].set_visible(False)

    fig.suptitle(f"Batch vs. Individual Inference ({n} images)", fontsize=13,
                 fontweight="bold")
    plt.tight_layout()
    fig.savefig("results/extra_batch.png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    print("  Figure saved → results/extra_batch.png")



# HELPER UTILITIES


def _print_table(headers: list, rows: list, title: str = ""):
    """Pretty-print a table to stdout."""
    if title:
        print(f"\n  ── {title} ──")
    if not rows:
        print("  (no data)")
        return
    widths = [max(len(h), max(len(str(r[i])) for r in rows))
              for i, h in enumerate(headers)]
    row_fmt = "  " + "  ".join(f"{{:<{w}}}" for w in widths)
    sep = "  " + "  ".join("-"*w for w in widths)
    print(row_fmt.format(*headers))
    print(sep)
    for row in rows:
        print(row_fmt.format(*row))



# MAIN


if __name__ == "__main__":
    print("SIS4 – Deep Learning Lab: Image Classification with Pretrained Models")
    print("="*70)

    print("\n[Setup] Downloading test images …")
    image_paths = download_images("images")

    t1_results  = task1_basic_classification(image_paths)
    t2_results  = task2_topk_predictions(image_paths)
    t3_results  = task3_model_comparison(image_paths)
    t4_errors   = task4_error_analysis(image_paths)
    extra_res   = extra_batch_processing(image_paths, n_total=10)

    all_results = {
        "task1": t1_results,
        "task2": t2_results,
        "task3": t3_results,
        "task4": t4_errors,
        "extra": extra_res,
    }
    with open("results/all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n" + "="*70)
    print("All tasks completed. Results saved to results/")
    print("="*70)