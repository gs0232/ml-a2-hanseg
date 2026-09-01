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
- Cell 8: xs, ys, cy, and cx -> what do they mean?
- Why is array shape so big?
- How can I explain Cochlea size of 108 voxels in a visual way? Is it really that small?
- Examine picture again to find real shape of Cochlea