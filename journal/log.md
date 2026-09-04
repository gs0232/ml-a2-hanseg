# Implementation log

## 2026-09-01
**Did:** 
- Reviewing quickstart for PyTorch (https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)
- and SimpleITK (https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/blob/main/Python/00_Setup.ipynb);
- Setup Google Colab with SimpleITK;
- Load HaN-Seg Dataset into Colab using !wget;
- Verify Dataset by using !md5sum (checksum)
- Unzip HaN-Seg Dataset
- Get shapes, sizes and numbers of data
- Determine Cochlea size
- Show image of Cochlea

Important notes:
- Array shape (slices, rows, columns): (202, 1024, 1024)
- Voxel spacing in mm (x, y, z): (0.5576171875, 0.5576171875, 2.0)
- Value range in Hounsfield units: -1000 to 3000
- Cochlea size: 108 voxels
- Cochlea size fraction: 0.00005% -> that is why mostly "background" will be classified
- CT-slice visualisation: When changing vmin=-500 and vmax=500 to -160 and 240, respectivley, the scan changes and shows soft tissue -> answers the question, why input channels in network are needed later
**Broke:** none
**Fixed by:** none
**Still unsure:**
- Cell 8: xs, ys, cy, and cx ... what do they mean? --> ys = row in which Cochlea is True; xs = column in which Cochlea is true; cy/cx = mean value which is the centre of the cochlea in that slice
- Why is array shape so big? --> depending on scanner
- How can I explain Cochlea size of 108 voxels in a visual way? Is it really that small? --> Each voxel is 0.5576 × 0.5576 × 2.0 = 0.62 mm³, so the cochlea is 67 mm³ which is a cube about 4 mm on a side.
- Examine picture again to find real shape of Cochlea

## 2026-09-04
**Did:** 
- Code load_case() function
- Split file finding from file reading: find_one() and case_files() -> case_files() can be tested without having the 4.9 GB dataset locally
- Add geometry check to load_case(): mask size and spacing must match the CT, otherwise ValueError
- Code resample() and window() in data.py
- Write 5 tests in tests/test_data.py, add empty conftest.py in repo root; all 5 pass with pytest -q
- Restructure a2_main.ipynb: clone repo into Colab and import from src/data.py instead of repeating the code in the cells
- Figure 2: soft-tissue window (40/400) vs bone window (400/1800) on the same slice

Important notes:
- Resampling to 1 mm isotropic: 1024x1024x202 -> 571x571x404 = 132 M voxels; at 0.5 mm it would be 1054 M voxels (8x) -> 0.5 mm only realistic on a small crop, not on the whole head
- Cochlea at 1 mm = ca. 67 voxels, ca. 4 voxels across; at 0.5 mm ca. 537 voxels, 8 across -> resolution decision still open
- Soft-tissue window: everything above 240 HU saturates to 1.0 -> all bone is flat white
- Bone window: soft tissue squashed into 0.22-0.31 -> nearly one single grey
- -> no single window shows parotid and cochlea, that is why the network gets 2 input channels

**Broke:** 
1. Bug in load_case() with "cases[0]" not being able to properly locate cases in case 2, 3, etc.
2. pytest -q: ModuleNotFoundError: No module named 'src'
3. Colab cell 1: ModuleNotFoundError: No module named 'imp' when running %load_ext autoreload

**Fixed by:** 
1. Changing "case[0]" to "case_dir" within the load_case() function
2. Adding an empty conftest.py in the repo root -> pytest then puts the repo root on the import path, not only tests/
3. Removing %load_ext autoreload -> Colab runs Python 3.13, "imp" was removed in Python 3.12, but Colab's IPython autoreload still imports it. Using importlib.reload(src.data) in the import cell instead

**Still unsure:** none