# PGCS project page

This directory is a dependency-free static site intended for GitHub Pages from the `/docs` folder on `main`. It presents the paper, method, selected author-provided examples, and citation. GitHub Pages does not execute the synthesis model.

## Image provenance

- `scene-01` through `scene-03`: matching `clear/QV` and `cloudy/QV` images in the authors' `Code/173/test_data` materials. The JPEGs here were resized for the web; each comparison uses the same numbered scene.
- `thickness-1` and `thickness-2`: increasing-thickness images `1_1`, `1_5`, `1_9`, `2_1`, `2_5`, and `2_9` from the authors' manuscript figure materials.
- `cloud-*`: generated examples from the authors' cloud-species analysis.
- `pgcs-framework.png`: the authors' framework figure.

The histograms' 80.06–91.89% range is from Fig. 7 of the published PGCS article. The main article is linked by DOI rather than mirroring the IEEE PDF here.

## Live demo gate

The planned online demo needs the existing CUDA/StyleGAN synthesis pipeline to run on a Hugging Face ZeroGPU Gradio Space with private model weights. Do not present it as available until a real GeoTIFF request succeeds on ZeroGPU. The current site intentionally shows saved examples and marks the live demo as pending.

The local source requires CUDA, GDAL and `libtiff`; the integrated Windows executable cannot be run on a Linux Space. ZeroGPU currently supports Gradio 4+, selected PyTorch 2.8+ releases, and Python 3.10.13 or 3.12.12. Porting and testing the source under those versions is required. The model file must stay in a private model repository and be read through a Space secret, not committed to this public repository.

For deployment, enable **Settings → Pages → Deploy from a branch → main → /docs** after pushing these files. The expected project URL is `https://liying-xu.github.io/PGCS/`.
