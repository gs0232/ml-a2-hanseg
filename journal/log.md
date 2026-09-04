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
**Broke:** 
1. Bug in load_case() with "cases[0]" not being able to properly locate cases in case 2, 3, etc.
**Fixed by:** 
1. Changing "case[0]" to "case_dir" within the load_case() function
**Still unsure:**
