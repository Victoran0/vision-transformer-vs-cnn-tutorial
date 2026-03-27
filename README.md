# Vision Transformers vs CNNs: A Comparative Tutorial

> **Why Self-Attention Can Extract Image Features Better Than Convolution**

This repository accompanies a machine learning tutorial that explains — with annotated figures, experimental results, and runnable code — how Vision Transformers (ViT) differ from Convolutional Neural Networks (CNNs) in the way they extract image features, and why self-attention can outperform convolution given sufficient data.

---

## 📁 Repository Contents

```
├── notebook.ipynb              # Main experiment notebook (ResNet-50 vs ViT fine-tuning)
├── ViT_vs_CNN_Tutorial.pdf    # Full written tutorial with figures
├── data_setup.py               # Helper: dataset download & DataLoader creation
├── engine.py                   # Helper: training & evaluation loop
├── utils.py                    # Helper: set_seeds, plot_loss_curves, timer
├── LICENSE
├── figures/
│   ├── fig1a_cnn_pipeline.png
│   ├── fig1b_vit_pipeline.png
│   ├── fig2_receptive.png
│   ├── fig3.1_attention.png
│   └── fig3.2_loss_comparison.png
│   ├── fig4_comparison.png
└── README.md
```

---

## 📖 Tutorial Overview

The tutorial is structured into six sections:

| Section | Title |
|---------|-------|
| 1 | Introduction — motivation and goals |
| 2 | How CNNs extract image features (local receptive fields, hierarchical learning) |
| 3 | How Vision Transformers process images (patch embedding, positional encoding, self-attention) |
| 3b | **Experimental results** — pre-trained ResNet-50 vs ViT fine-tuned on Food-101 |
| 4 | Why self-attention can be better than convolution |
| 5 | Discussion — when CNNs still win |
| 6 | Conclusion and references |

---

## 🧪 Running the Notebook

### What the notebook does

The notebook fine-tunes two pre-trained models on a **225-image subset of Food-101** (pizza, steak, sushi) for 25 epochs and plots their training and test loss curves:

- **ResNet-50** (`torchvision.models.resnet50`) — convolutional backbone, only the final fully-connected head is unfrozen for training.
- **ViT-B/16** (`torchvision.models.vit_b_16`) — Vision Transformer, patch projection and encoder are frozen; only the classifier head is trained.

### Prerequisites

- Python 3.9+
- `pip` or a virtual environment (recommended)

### Step 1 — Clone the repository

```bash
git clone https://github.com/victoran0/vision-transformer-vs-cnn-tutorial.git
cd <your-repo>
```

### Step 2 — Install dependencies

#### ✅ If you have a CUDA-enabled GPU (recommended)

The notebook was originally run with CUDA 12.8. Install the GPU build of PyTorch:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install torchinfo tqdm jupyter
```

#### 🖥️ If you do NOT have a GPU (CPU-only)

Install the standard CPU build instead:

```bash
pip install torch torchvision
pip install torchinfo tqdm jupyter
```

> **⚠️ Important — CPU training time warning**
>
> Training on CPU is significantly slower than on a CUDA-enabled GPU. The table below gives approximate estimates for this notebook's setup (25 epochs, 225 images, batch size 32):
>
> | Hardware | Estimated time per epoch | Total (25 epochs) |
> |----------|--------------------------|-------------------|
> | CUDA GPU (e.g. RTX 3080) | ~5–15 seconds | ~5–10 minutes |
> | Apple Silicon (MPS) | ~20–40 seconds | ~15–20 minutes |
> | CPU only (modern laptop) | ~3–8 minutes | ~1.5–4 hours |
>
> **Practical tips for CPU users:**
> - Reduce the number of epochs: change `NUM_EPOCHS = 25` to `NUM_EPOCHS = 5` to get indicative results quickly.
> - Reduce batch size to `BATCH_SIZE = 8` if you run into memory issues.
> - The CPU build still produces valid results — training will converge, just more slowly.
> - Consider using [Google Colab](https://colab.research.google.com/) (free T4 GPU) or [Kaggle Notebooks](https://www.kaggle.com/code) (free P100 GPU) to run at full speed.

The notebook automatically detects your hardware:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

No code changes are needed — it will fall back to CPU automatically if no GPU is found.

### Step 3 — Launch the notebook

```bash
jupyter notebook notebook.ipynb
```

Run all cells from top to bottom. The dataset (~3 MB) is downloaded automatically in Cell 8 if not already present.

---

## ♿ Accessibility

This tutorial was designed so that **students with disabilities can engage with all content fully**. Two categories of disability are specifically addressed.

### 👁️ Colour Blindness — Visual Accessibility

All figures in this tutorial use the **Okabe-Ito colour palette**, a scientifically validated set endorsed by Nature and Science journals and designed to be distinguishable across all common forms of colour vision deficiency, including:

- **Deuteranopia** (red-green, most common — affects ~8% of males)
- **Protanopia** (red-green, less common)
- **Tritanopia** (blue-yellow, rare)

Beyond colour, every figure uses **at least two additional redundant visual cues** so that meaning is never conveyed by colour alone:

| Figure | Colour-safe palette | Redundant cue 1 | Redundant cue 2 |
|--------|--------------------|--------------------|-----------------|
| Fig 1a — CNN pipeline | Okabe-Ito blue (#0072B2) | `//` hatch pattern on boxes | Text labels inside each box |
| Fig 1b — ViT pipeline | Okabe-Ito amber (#E69F00) | `xx` hatch pattern on boxes | Text labels inside each box |
| Fig 2 — Receptive fields | Blue (CNN) vs amber (ViT) | `//` vs `xx` hatch + thick border | `●` marker (CNN active), `★` marker (ViT centre) |
| Fig 3.1 — Self-attention | Green/sky-blue/vermillion for Q/K/V | Named labels (Q, K, V) on every box | Legend strip with colour + pattern |
| Fig 3.2 — Loss curves | Original matplotlib yellow/blue lines | Annotated callout boxes with exact values | Separate subplot per model with title |
| Fig 4 — Bar chart | Blue (CNN) vs amber (ViT) | `//` vs `xx` hatch on bars | Numeric value labels above each bar |

To verify the figures yourself under simulated colour blindness, tools like [Coblis](https://www.color-blindness.com/coblis-color-blindness-simulator/) or the [Chromatic Vision Simulator](https://asada.website/chromavision/) can be used.

### 🔊 Screen Reader Support — Alt Text

All figures embedded in `ViT_vs_CNN_Tutorial.docx` should have descriptive alt text added so screen readers can convey the content to visually impaired students. If you open the docx, right-click each image → **Edit Alt Text** and paste the descriptions below.

**Figure 1a — CNN Feature Extraction Pipeline**
> Flowchart showing the CNN feature extraction pipeline. Five boxes connected by arrows from left to right: Input Image, Conv Layer 1 (Edges), Conv Layer 2 (Textures), Conv Layer N (Objects), Feature Map. Arrows between boxes are labelled: 3x3 kernels, 3x3 kernels, stacked layers, global pooling. All boxes are blue with diagonal hatch lines.

**Figure 1b — Vision Transformer Pipeline**
> Flowchart showing the Vision Transformer pipeline. Five boxes connected by arrows from left to right: Input Image, Patch Split 16x16, Patch Embedding plus Positional Encoding, Self-Attention Layers, Class Token to Label. Arrows between boxes are labelled: to tokens, linear projection, all-to-all attention, MLP head. All boxes are amber with cross hatch lines.

**Figure 2 — Local vs Global Feature Extraction**
> Two 7-by-7 grids side by side. Left grid labelled CNN: Local Receptive Field. A 3-by-3 region of cells is highlighted in blue with diagonal hatching and dot symbols, surrounded by a thick blue border, showing the limited area a single convolutional kernel sees. Right grid labelled ViT: Global Self-Attention. The centre cell is highlighted in amber with cross hatching and a star symbol. Amber arrows radiate outward from the centre cell to every other cell in the grid, illustrating that one patch attends to all other patches simultaneously.

**Figure 3 — Self-Attention Mechanism**
> Diagram showing the self-attention mechanism. Top row: four coloured patch boxes labelled Patch 1 face, Patch 2 background, Patch 3 ear, Patch 4 background. Below them: three boxes labelled Q Query in green, K Key in sky blue, V Value in vermillion, connected by upward arrows. To the right: a Softmax Scores box in dark navy connected by a horizontal arrow, then an Output Context box in purple connected by a downward arrow. Below all boxes: the attention formula: Attention(Q,K,V) equals softmax of Q times K-transpose divided by square root of dk, times V.

**Figure 4 — Qualitative Comparison Bar Chart**
> Grouped bar chart comparing CNN (ResNet) in blue with diagonal hatching against Vision Transformer in amber with cross hatching across five dimensions: Global Context (CNN 2, ViT 5), Adaptive Weights (CNN 1, ViT 5), Data Efficiency (CNN 5, ViT 2), Compute Cost (CNN 4, ViT 2), Small Datasets (CNN 5, ViT 2). Scores range from 1 (low) to 5 (high). Numeric values are labelled above each bar.

**Figure 5 — Training Loss Comparison**
> Two line graphs side by side. Left graph titled ResNet-18 Pre-trained, Food-101 Subset. Shows training loss (yellow line) declining from 1.07 to approximately 0.25 over 25 epochs, and test loss (blue line) declining more slowly and plateauing around 0.50. An annotation points to the training loss final value of approximately 0.25, and another points to the test loss plateau at approximately 0.50. Right graph titled Vision Transformer Pre-trained, Food-101 Subset. Shows training loss (yellow line) dropping steeply in the first 2 epochs from 0.47 to approximately 0.07 and remaining low. Test loss (blue line) stays stable between 0.19 and 0.28 throughout training. An annotation points to the training loss value of approximately 0.07, and another points to the stable test loss around 0.20.

---

## 📚 References

- Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. (2020). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* arXiv:2010.11929.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition.* CVPR 2016.
- Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). *Attention Is All You Need.* NeurIPS 2017.
- Bourke, D. (2023). *PyTorch Deep Learning — Zero to Mastery Course.* GitHub: [mrdbourke/pytorch-deep-learning](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/04_pytorch_custom_datasets.ipynb).

---

## 📝 License
MIT   
This tutorial and its materials are shared for educational purposes.
