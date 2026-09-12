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
- Pixel accuracy = 0.99811 -> PROVES THAT ACCURACY IS A MEANINGLESS METRIC FOR THIS TASK AND THE LOSS FUNCTION HAS TO BE DESIGN AND NOT PICKED because 99.8% is background anyway
- All rows are dice = 0.0, hd95_mm = NaN, sdice_1mm = NaN
    - Except: bone_threshold Cochlea_L  dice = 0.0003, hd95_mm = 228.2187, sdice_1mm = 0.001
    - --> this is because the function has an attribute csl = 1, so it only looks for class 1, which is the left Cochlea; The numbers represent a weak result but they give a result. However, this is a demonstration of why intensity alone cannot work, not as a serious competitor and therefore the model has to be trained.
    - Dice = Is it classified right (0/1)
    - Surface Dice = How much of the outline looks like the real outline within tolerance (0-100% and a set tolerance) -> so sdice_1mm = 0.001 means that only 0.1% of the outline was matched within 1mm of tolerance
    - hd95_mm = 228.2187 --> 95th percentile of boundary distances
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
- Rerun a2_main with 42 cases
- Update preprocessing_1.csv and add crop_losses.csv
- Add baseline results from 9 test cases
- Build loader and model
- Run first train run in a2_main
- Evaluate metrics on 9 patients

Important Notes:
- model.py owns a function that has 7.8 million parameters that turns one slice into 5 numbers per pixel
- losses.py computes the score of how the predicition matches the original which then can be differentiated and set a foundation on how the parameters should be adjusted
- loader.py + train.py feeds batch by batch containing slices; runs the adjust-and-repeat loop
- eps is a number put on top and bottom of Dice fraction to prevent division by zero
    - eps should stay at 1.0 because softmax never is exactly 0 (a structure that's absent from both the truth and the prediction still accumulates about 0.0000001 of "prediction mass" across 65,000 pixels.). If eps would fall below a threshold, the model would extremly punish a perfectly correct prediction
- Optimizer: AdamW (steps cautiously where gradient is erratic)
- Crop losses in cases 21, 30, 32, 35, 37, 39
- loader and model:
    - 3538 training slices (selected)
    - 3078 validation slices (all of them)
    - class counts in training:
        - background   230,568,530   99.44026 %
        - Cochlea_L          3,500    0.00151 %
        - Cochlea_R          3,732    0.00161 %
        - Parotid_L        646,810    0.27896 %
        - Parotid_R        643,796    0.27766 %
        - parotid : cochlea = 185 : 1

    - 7,762,885 parameters
    - shape check: (2, 5, 256, 256)
- first training run: see experiments/train_run_1.csv
    - Parotid_L 0.814, Parotid_R 0.823 against a baseline of 0.000
    - Cochlea 0.000 on all 9 test patients, 18 complete misses of 36
- first 3d metrics on 9 patients: see experiments/metrics_1.csv
    - Dice = 0.82 is strong
    - sDice 1mm = 0.58 is ok
    - big difference between HD95 and HDMAX

Limitation found:
- Outline for Cochlea in dataset grow bigger towards later cases (case 1-20 = median 101 voxels; case 21-42 = median 196 voxels). Therefore the split is uneven (train median 160.5, val median 136.5, test median 198.5). No re-splitting because should stay random. Parotid does not significantly change. Possible reaons for bigger Cochlea voxels might be change of protocols after case 20 or different scanner parameters (check at next unzip) that would cause higher resolution. "The ground truth is itself inconsistent, so Dice has a ceiling below 1 that has nothing to do with the model" - claude

                      cases 01-20   cases 21-42   ratio   Mann-Whitney p
cochlea (per side)            101           196    1.94         2.3e-08
parotid (per side)         25,845        28,277    1.09         6.2e-02
cochlea / parotid           4.51          6.19     1.37         1.1e-02

Key points for Criterion C in report:
- Dice counts overlap; surgery needs boundary distance in millimetres
- Dice is symmetric; surgical error costs are not
- The ground truth is itself inconsistent, so Dice has a ceiling below 1 that has nothing to do with the model

**Broke:**
- Cochlea can't be found
**Fixed by:**
- Not sure yet, possibly with eps. --> No, cross-entropy finds Cochlea (see 2026-09-10)

**Still unsure:**
- Why 7.8 MILLION parameters? --> see claude table exlpanation with 3x3 * 2 * 16 + ...
- one slice into 5 numbers per pixel? --> for background and the four structures; is then turned into probabilities of each class by softmax -> [0.9 0.02 0.04 0.0 0.4] = background
- Can we not just download more cases instead of mirroring them? --> Yes, did that


## 2026-09-10
**Did:**
- Plot training data from first train run + code plots for the loss functions and dice validation of the big run
- Renumbered a2_main
- Start 5 more training runs. First four have different loss strategy
- Recover lost csv files from cache
- Build plots from notebook output log (the one's that have not been lost)
- Run test of compound loss to find out ratio between cross-entropy and dice

Important Notes:
- Training runs and what changed:
    - 0 - Compound loss: done yesterday
    - 1 - **cross entropy loss: found Cochlea AND Parotid**
    - 2 - dice loss: region-wise, imbalance-aware, unstable on tiny structure
    - 3 - compound loss: standard combo
    - 4 - tversky loss: matches my objective
    - 5 - change training mix: 36% of slices contain a cochlea (before: 8.6%)
- history csv files are reconstructed from printed notebook output log --> see comments in RECOVERED.md for citation in journal
- Compound = ce + dice --> over 20 training batches, at the compound run's best weights
    - cross-entropy term  0.00686
    - soft Dice term      0.25366
    - compound total      0.26052
    - cross-entropy is 2.63% of the compound loss

**Broke:** 
1. plot design: center subheading wasn't working because x = 0.0
2. Runtime died during run 4/5 at epoch 25
3. I ran out of Google Colab T4 limit
**Fixed by:**
1. plot design: changed x = 0.5
2. adapt code so csv gets safed earlier in the process + cell 27 as a recovery for the lost csv files
3. Used CPU for csv recovery

**Still unsure:**
- Cell 28 and its purpose or how to interpret it. It is a test made by Claude but I don't know what the test is for

## 2026-09-11
**Did:**
- No model work


## 2026-09-12
**Did:**
- Answer some knowledge gaps
- Rerun training run 4 and 5 after checking if files for 1-3 exist

Important Notes:
- Cell 29 Output - Gradient Norms:
gradient norm sent back into the logits, averaged over 5 batches
(at the compound run's best weights — the state where cochlea = 0)

term             background    Cochlea_L    Cochlea_R    Parotid_L    Parotid_R

cross-entropy       0.00005      0.00000      0.00000      0.00003      0.00003
soft Dice           0.00041      0.00000      0.00000      0.00032      0.00026

ratio, cross-entropy / soft Dice, per class:
  background       0.12x
  Cochlea_L        2.07x
  Cochlea_R       10.53x
  Parotid_L        0.10x
  Parotid_R        0.13x


**Broke:**
- Plot for validation mean dice: legend!
**Fixed by:**
- not yet fixed!

**Still unsure:**
- Norm Gradient