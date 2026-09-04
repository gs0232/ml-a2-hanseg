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

## 2026-09-04 - Claude, data.py and test_data.py
**Got:** Code snippets to use for functions in data.py and test_data.py; instructions on how to use colab + github
**How I used it:** Inserted and typed in code snippets + changed what had to be changed (file names for example)
**Verified by:** Researched what unknown code means and run the code