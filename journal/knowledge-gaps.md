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
- Am I doing maxpooling or batch normalisation?

## 2026-09-10
- First training run plot shows a drop in 2nd epoch in Dice validation. Why? --> Problem of Batch/Group Normalisation because after 1 epoch averages are not right (VERIFY IF I SAID IT RIGHTT)