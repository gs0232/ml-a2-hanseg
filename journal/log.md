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
- Split file-finding from file-reading: find_one() and case_files() -> case_files() can be tested without having the 4.9 GB dataset locally
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

**Still unsure:** 
- resampling function: What is it used for?

## 2026-09-06
**Did:**
- Recap of already existing function and diving deeper into resampling
- code data.py/head_centre to find center of head in every image/slice
- code data.py/crop_to to extract head out of every slice
- code data.py/build_label to assign background (0) or one of the four binary masks classes (1-4) to a voxel (Coch R, Coch L, Paro R, Paro L)
- code data.py/select_slices to keep track of the indices of slices that acutally hold one of the four structures + a few random empty ones
- code data.py/preprocess_case
- add tests/test_prep.py to test the previous coded functions
- restructure a2_main.ipynb and add cache for easier future usage
- Preprocess and cache 20 cases, back up to Drive (284 MB)

Important Notes:
- Resampling makes cubes out of the boxes (from scanner)
- Interpolation decides if a voxel cube is then 0 or 1; shouldn't be a float because only classes can be assigned
- Recap: One epoch works like this: run one batch forward -> compute loss -> compute gradients backwards -> optimizer decides step size for each weight
- **HYPERPARAMETERS**
    - Spacing/Resolution: 1 mm - isotropic spacing (1 x 1 x 1 mm)
    - Crop: 256 mm - smallest area that holds both Parotid and Cochlea
    - Channels: 2 - ch0 for soft tissue window and ch1 for bone window
    - Keep: 1 - means that one empty slice per full slice (slice with structure) is kept -> should only be in training set (has to be implemented)
- Output from preprocessing available in expriments/preprocessing_1.csv
- Wrong interpolator: Cochlea_L 66 voxels with is_mask=True vs 38 with is_mask=False -> linear interpolation destroys 42% of the cochlea
- Class balance over 20 cases: cochlea 0.00089%, parotid 0.191%, background 99.808%
- Median parotid is 256x the median cochlea
- Cochlea 56 to 270 voxels between patients (4.8x) -> expect high variance in per-case Dice
- Slices per case 348 to 603

**Broke:** 
1. preprocess_case returned 2 values with keep_all=True and 3 without -> caching cell could not unpack it
2. git add: fatal: Unable to create .git/index.lock: File exists
3. UserWarning on 4 of 20 cases: crop lost label voxels (case_05 0.03%, case_12 0.45%, case_18 2.52%, case_20 0.68%)
**Fixed by:** 
1. Always return 3 values; with keep_all=True, keep = np.arange(len(lab_c))
2. rm -f .git/index.lock (leftover lock, no git process running)
3. Open -> need to check which structure the crop clips before deciding whether to re-cache

**Still unsure:** 
- what is the "out" in crop_to --> gives us information where the image was cropped and where it is missing rows/columns to fill them up with pads
- what are the four binary mask classes? Why four? --> Cochlea R, Cochlea L, Parotid R, Parotid L
- what is select_slices for? Why do we have to keep empty slices --> because model would otherwise think that there is ALWAYS a structure in an image
- why "del" commands in data.py/preprocess_case --> keeps memory down

## 2026-09-07
**Did:**
- Looked at the lost cases to identify the problem
- Compared the dice and hd metrics
- Split the datasets in training, test, and validation (12, 4, 4)
- Running the split and predicition of background and bone structure on test dataset (experiments/split-pixel_accuracy.csv) to evaluate the metrics

Important Notes:
- There are 4 lost cases (5, 12, 18, 20) where some of the structure was cropped. It was always the Parotids, never the Cochlea that got cropped. Most of the cropping was below 1.5% except for case 18, where Parotid_L was lost with 4.4%. 
- When looking at thin spurs, both dice and hd95 would miss it. hd_max would see it
- Pixel accuracy = 0.99811 -> PROOFS THAT ACCURACY IS A MEANINGLESS METRIC FOR THIS TASK AND THE LOSS FUNCTION HAS TO BE DESIGN AND NOT PICKED because 99.8% is background anyway
- All rows are dice = 0.0, hd95_mm = NaN, sdice_1mm = NaN
    - Except: bone_threshold Cochlea_L  dice = 0.0003, hd95_mm = 228.2187, sdice_1mm = 0.001
    - --> this is because the function has an attribute csl = 1, so it only looks for class 1, which is the left Cochlea; The numbers represent a weak result but they give a result. However, this is a demonstration of why intensity alone cannot work, not as a serious competitor and therefore the model has to be trained.
    - Dice = Is it classified right (0/1)
    - Surface Dice = How much of the outline looks like the real outline within tolerance (0-100% and a set tolerance) -> so sdice_1mm = 0.001 means that only 0.1% of the outline was matched within 1mm of tolerance
    - hd95_mm = 228.2187 greatest distance from the actual right outline (hard to explain)
- The baseline was set with this day

**Broke:** none
**Fixed by:** none

**Still unsure:** 
- what to do with the hd95 missing the thin spur
- What does the row information mean, that differs from the other rows when predicting the pixel accuracy --> see Important Notes

## 2026-09-08
**Did:**
- Recap of results from the day before
- Setting goal for today: loss, optimizer, and hyperparameters
- Mirror cases in loader for more data
- Define Batch = 8 slices, Epochs = 30, Learning rate = 0.001
- Add cases from HaN-Seg to have the following train, val, and test set (25, 8, 9)

Important Notes:
- model.py owns a function that has 7.8 million parameters that turns one slice into 5 numbers per pixel
- losses.py computes the score of how the predicition matches the original which then can be differentiated and set a foundation on how the parameters should be adjusted
- loader.py + train.py feeds batch by batch containing slices; runs the adjust-and-repeat loop
- eps is a number put on top and bottom of Dice fraction to prevent division by zero
    - eps should stay at 1.0 because softmax never is exactly 0 (a structure that's absent from both the truth and the prediction still accumulates about 0.0000001 of "prediction mass" across 65,000 pixels.). If eps would fall below a threshold, the model would extremly punish a perfectly correct prediction
- Optimizer: AdamW (steps cautiously where gradient is erratic)

**Broke:** none
**Fixed by:** none

**Still unsure:**
- Why 7.8 MILLION parameters?
- one slice into 5 numbers per pixel? --> for background and the four structures; is then turned into probabilities of each class by softmax -> [0.9 0.02 0.04 0.0 0.4] = background
- Can we not just download more cases instead of mirroring them?