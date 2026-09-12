What kind of CNN/System am I implementing?
- U-Net? which is semantic segmentaion to classify individual pixels

## 2026-09-07
- 4 lost cases due to cropping of structured voxels. However, only Parotid has been cropped, not Cochlea. Worst case is with loss of 4.4%
- Both, Dice and HD95, would miss a thin spur. Only HD Max would see it. But HD Max is too noisy to use. What am i supposed to use? How should I handle this?

## 2026-09-08
- Tversky and Dice difference/comparison/similarity
- eps is still a little bit unclear regarding its impact on the loss
- What is a softmax function?
    - It is an activation function for systems with more than 2 classes (multiclass)
    - Since we have 5 classes, softmax is used to create the probabilities that add up to 1
    - Sigmoid would be an activation function for systems with only 2 classes (binary)

## 2026-09-09
- Am I doing maxpooling or batch normalisation? --> Both, MaxPool2d halves the resolution (image size) and BatchNorm2d rescales the activations so training stays stable (changes number ranges)

## 2026-09-10
- First training run plot shows a drop in 2nd epoch in Dice validation. Why
    - Problem of Batch/Group Normalisation because during training BatchNorm normalises using the current batch's statistics; during validation the model is in eval() mode and uses running averages accumulated; After one epoch those averages haven't converged, so validation is computed with the wrong normalisation;
- How do we know that 36% of slices contain cochlea in the training run with the other training mix? So is it not randomized or is it just the numbers we know?
    - First thought: we have a table with cases and their number of slices with cochlea and parotid. That's probably where the 36% come from. But where does the new training mix get made?
    - Answer: directly measured with train_loader.dataset.fraction_containing((1, 2)) then the new mix is made in SliceDataset.__init__ where slices containing a rare class get their index repeated 

## 2026-09-12
- Am I doing Batch or Group normalisation?