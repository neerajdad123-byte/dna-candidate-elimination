# Class-DNA Candidate Elimination for Efficient Neural Network Inference

## What Is Happening
Standard neural networks compute probabilities against **every class** 
for every input — even impossible ones. A clearly written "1" should 
never waste computation comparing against "0" or "8".

This work eliminates statistically impossible candidates **before** 
full inference using lightweight prototype matching.

## Key Idea
1. Extract mean feature prototype (DNA) for each class from training data
2. At inference, match input against all class DNAs using cosine similarity
3. Keep only top-K candidates, mask the rest
4. Run full network only for surviving candidates
5. Exit early if confidence exceeds 90%

## Results on MNIST (10,000 test images)

| TOP_K | Computation Reduction | Accuracy Drop | Early Exits |
|-------|----------------------|---------------|-------------|
|   3   |        70%           |    2.07%      |   85.3%     |
|   4   |        60%           |    1.08%      |   83.8%     |
|   5   |        50%           |    0.63%      |   82.5%     |

**Sweet spot: TOP_K=5 → 50% less computation, only 0.63% accuracy drop**

## Requirements
```
torch, torchvision, numpy, matplotlib
```

## Usage
```bash
python your_network.py
```

## Author
Neeraj — Age 17 — Independent Research
