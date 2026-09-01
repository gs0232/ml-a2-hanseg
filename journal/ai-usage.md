# Use of AI tools

## 2026-08-27 — Claude, project scoping
**Asked:** whether a 3D Gaussian Splatting project was feasible for this assignment.
**Got:** a recommendation against it (CUDA build vs the self-contained-Colab
requirement) and an alternative: OAR segmentation on HaN-Seg.
**My judgement:** accepted. I checked the R2-Gaussian repo myself and confirmed
it pins PyTorch 2.1.2 / CUDA 11.8 and builds TIGRE from source.
**Not accepted:** the original framing motivated the task by radiotherapy dose.
My interest is surgical planning, so I rewrote the motivation.

## 2026-09-01 - Claude, setup
**Got:** Step-by-step guide on how to set up Google Colab and the Dataset
**How I used it:** set up the folder framework and code scaffold myself using online ressources (PyTorch Quickstart and SimpleITK repo); asked Claude how to step-by-step get the configurations right for the Google Colab.
**Verified by:** running the code.