# FGVC-Aircraft Benchmark Experiment Log
**Model Architecture:** ConvNeXt V2 Tiny  
**Dataset:** FGVC-Aircraft (3,334 train / 3,333 val samples)  
**Target Classes:** 100 Variants, 70 Families, 30 Manufacturers  
**Hardware Environment:** AMD Ryzen 5 5625U CPU (6 Cores / 12 Threads)

---

## 1. Executive Summary & Experiment Tracker

This log tracks the iterative development of an automated hierarchical aircraft classification pipeline. The model transitions from baseline setup through linear classifier head alignment (Phase 2) to full backbone fine-tuning (Phase 3).

| Run ID | Phase / Strategy | Primary Changes Made | Justification & Objectives | Train Acc (Variant) | Val Acc (Variant / Family / Manu) | Status / Decision |
| :---: | :--- | :--- | :--- | :---: | :---: | :--- |
| **`v1.0`** | **Phase 1:** Head-Only Training | Backbone frozen (`requires_grad=False`). Trained final linear head for 5 epochs using `AdamW` (`lr=1e-3`, `weight_decay=1e-2`). | Adapt new classifier head weights to ConvNeXt V2's pre-trained ImageNet feature space without ruining backbone representations. | **52.22%** | **34.38% / 44.28% / 56.74%** | Need to unfreeze backbone tuning to improve accuracy |
| **`v2.0`** | **Phase 2:** Full Fine-Tuning | Increased epoch number (5->10). Unfreeze backbone (`requires_grad=True`). Set lower learning rate at (`lr=1e-5`) for initial backbone layers to (`lr=1e-4`) for classification head layers. Added `CosineAnnealingLR` for LR tuning per epoch | Adapt deep convolutional features to capture aviation-specific details (engine placement, winglets, door configurations). | **100%** | **80.80% / 88.69% / 92.35%** | Delay/prevent 100% training saturation to try improve model's understanding of aircraft differences |
| **`v3.0`** | **Phase 3:** Full Fine-Tuning | Increased epoch number (5->10). Unfreeze backbone (`requires_grad=True`). Set lower learning rate at (`lr=1e-5`) for initial backbone layers to (`lr=1e-4`) for classification head layers. Added `CosineAnnealingLR` for LR tuning per epoch | Adapt deep convolutional features to capture aviation-specific details (engine placement, winglets, door configurations). | **100%** | **80.80% / 88.69% / 92.35%** | Delay/prevent 100% training saturation to try improve model's understanding of aircraft differences |

---

## 2. Phase 1 Experiment Deep-Dive (`v1.0`)

### 2.1 Configuration Parameters
* **Backbone:** ConvNeXt V2 Tiny (Pre-trained on ImageNet-1K)
* **Backbone Status:** Frozen (`requires_grad = False`)
* **Classifier Head:** Linear layer ($768 \rightarrow 100$)
* **Optimizer:** `AdamW(lr=1e-3, weight_decay=1e-2)` (Filtering parameters where `requires_grad=True`)
* **Loss Function:** `nn.CrossEntropyLoss()`
* **Batch Size:** 16 (12 logical threads, CPU execution)
* **Epochs:** 5

### 2.2 Epoch-by-Epoch Progress Log
During Phase 1 training, loss and training accuracy progressed steadily across 209 iterations per epoch:

```text
Epoch 1: Loss = 3.6800 | Training Accuracy = 15.57%
Epoch 2: Loss = 2.7480 | Training Accuracy = 31.55%
Epoch 3: Loss = 2.3615 | Training Accuracy = 39.38%
Epoch 4: Loss = 2.0636 | Training Accuracy = 47.09%
Epoch 5: Loss = 1.8636 | Training Accuracy = 52.22%
```

### 2.3 Quantitative Validation Results
Validation evaluation was executed across 3,333 unseen images using vectorized tensor lookup maps (`v2f_tensor`, `v2m_tensor`) to measure accuracy across all three benchmark taxonomy levels:

* **Variant Accuracy (100 Classes):** `34.38%`
* **Family Accuracy (70 Classes):** `44.28%`
* **Manufacturer Accuracy (30 Classes):** `56.74%`

### 2.4 Validation Hierarchy Verification
The hierarchical accuracy scaling holds strictly to expectations:
$$\text{Variant Accuracy (34.38\%)} < \text{Family Accuracy (44.28\%)} < \text{Manufacturer Accuracy (56.74\%)}$$

* **Reasoning:** Fine-grained variant misclassifications (e.g., confusing a *Boeing 737-800* with a *Boeing 737-900*) frequently still resolve to the correct parent family (*Boeing 737*) and manufacturer (*Boeing*). This confirms the accuracy lookup maps and index alignments across dataset splits are mathematically sound.

### 2.5 Analysis of the Generalization Gap (~52.22% Train vs ~34.38% Val)
* **Diagnosis:** A 17.74% performance gap between training accuracy and validation accuracy indicates that a single linear classification layer on top of static ImageNet features has reached its upper capability limit.
* **Root Cause:** ImageNet pre-training focuses on high-level shape separation (e.g., distinguishing distinct categories like dogs from cars). Fine-grained aircraft recognition requires learning subtle visual differences (e.g., window counts, winglet angles, engine cowl dimensions) that frozen ImageNet features cannot fully represent.

---

## 3. Phase 2 Full Fine-Tuning (`v2.0`)

### 3.1 Objective
Unfreeze all backbone parameters and fine-tune the entire network to extract aviation-specific visual features, targeting **70%+ Variant Accuracy**.

### 3.2 Implementation Requirements

1. **Unfreeze Backbone Parameters:**
   ```python
   for param in model.parameters():
       param.requires_grad = True
   ```

2. **Configure Lower Learning Rate:**
   Reduce the learning rate from $10^{-3}$ to $10^{-5}$ to prevent destructive gradient updates from overwriting pre-trained weights (*catastrophic forgetting*).

3. **Apply Differential Learning Rates:**
   Assign a lower learning rate ($10^{-5}$) to early backbone layers and a slightly higher rate ($10^{-4}$) to the classification head.
   ```python
   optimiser = optim.AdamW(
    [  
        {"params": backbone_params, "lr": 1e-5},
        {"params": head_params, "lr": 1e-4}
    ],
    weight_decay=1e-2
   )   
   ```

4. **Implement Learning Rate Scheduler:**
   Implement `CosineAnnealingLR` to protect backbone from over-adjusting over 10 epochs, lower chance of overfitting occuring. 
   ```python
   scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=10, eta_min=1e-6)
   ```
   `scheduler.step()` called each epoch to implement

5. **Increase Epoch Number**
   Increase Epochs from 5 -> 10
      * This is needed for the backbone parameters to adapt further to aircraft features, whilst learning rate tapers up.

### 3.3 Configuration Parameters
* **Backbone:** ConvNeXt V2 Tiny (Pre-trained on ImageNet-1K)
* **Backbone Status:** Unfrozen (`requires_grad = True`)
* **Classifier Head:** Linear layer ($768 \rightarrow 100$)
* **Optimizer:** `AdamW(backbone lr=1e-5, head lr=1e-4, weight_decay=1e-2)` (Filtering parameters where `requires_grad=True`)
* **Learning Rate Scheduler *(NEW)*:** `CosineAnnealingLR(T_max=10, eta_min=1e-6)`
* **Loss Function:** `nn.CrossEntropyLoss()`
* **Batch Size:** 16 (`num_workers=2`, `pin_memory=True`, GPU execution)
* **Hardware Accelerator *(NEW)*:** Google Colab NVIDIA Tesla T4 GPU
* **Epochs:** 10

### 3.4 Epoch-by-Epoch Progress Log
During Phase 2 training, loss decreases and training accuracy rapidly increases for the first 4 epochs until it flatenned at 100% training accuracy by epoch 7:

```text
Epoch 1: Loss = 3.4842 | Training Accuracy = 19.29%
Epoch 2: Loss = 1.4667 | Training Accuracy = 62.06%
Epoch 3: Loss = 0.5470 | Training Accuracy = 87.10%
Epoch 4: Loss = 0.1955 | Training Accuracy = 96.97%
Epoch 5: Loss = 0.0779 | Training Accuracy = 99.31%
Epoch 6: Loss = 0.0364 | Training Accuracy = 99.97%
Epoch 7: Loss = 0.0217 | Training Accuracy = 100.00%
Epoch 8: Loss = 0.0170 | Training Accuracy = 100.00%
Epoch 9: Loss = 0.0146 | Training Accuracy = 100.00%
Epoch 10: Loss = 0.0136 | Training Accuracy = 100.00%
```
### 3.5 Quantitative Validation Results
Validation evaluation was executed across 3,333 unseen images using vectorized tensor lookup maps (`v2f_tensor`, `v2m_tensor`) to measure accuracy across all three benchmark taxonomy levels:

* **Variant Accuracy (100 Classes):** `80.80%`
* **Family Accuracy (70 Classes):** `88.69%`
* **Manufacturer Accuracy (30 Classes):** `92.35%`

### 3.6 Analysis of results
* Although model begun to overfit training dataset by the 4th epoch, the model still succesfully exceeded 70% target for variant accuracy set, acheiving accuracy of 80.80%.
* The hierarchical performance remained intact as expected:
$$\text{Variant Accuracy (80.80\%)} < \text{Family Accuracy (88.69\%)} < \text{Manufacturer Accuracy (92.35\%)}$$
   * This suggests that although model did start to overfit the training data, the backbone did actually develop understanding of structural aircraft features, with a strong understanding of difference between different families of aircraft.
* Needed to use virtual GPU training instead of native CPU due to higher compute needs to adjust backbone.

---

## 4. Phase 3 Mitigation of Model Saturation via Data Augmentation (`v3.0`)

### 4.1 Objective
Prevent training saturation and augment the training images to encourage deeper understanding of different smaller plane properties (e.g. winglets), increase input resolution to capture differences within an aircraft family. 

### 4.2 Implementation Requirements

1. **Input Resolution Increase:**
   Update sizing from 224x224 &rarr; 288x288

2. **Spatial Cutout:**
   Implement random errasing to force model to look at alternative regions of planes, to prevent overfitting.
   ```python
   v2.RandomErasing(p=0.25, scale=(0.02, 0.2), value='random')
   ```

3. **Image Noise**
   Reduce reliance on background sky colour and lighting through implementing changes in photo's lighting.
   ```python
   v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)
   ```

4. **Training Dynamics:**
   Increase `weight_decay` in `AdamW` from `1e-2` &rarr; `0.05` to penalise large weight magnitudes and delay the training saturation 

5. **Stochastic Depth**
   Instantiate `timm` model with `drop_path_rate=0.1` to randomly drop residual paths during forward pass. This will reduce reliance on previous layers being active to encourage layers to produce useful, independency.

### 4.3 Configuration Parameters
* **Backbone:** ConvNeXt V2 Tiny (Pre-trained on ImageNet-1K)
* **Backbone Status:** Unfrozen (`requires_grad = True`)
* **Classifier Head:** Linear layer ($768 \rightarrow 100$)
* **Optimizer:** `AdamW(backbone lr=1e-5, head lr=1e-4, weight_decay=0.05)` (Filtering parameters where `requires_grad=True`)
* **Learning Rate Scheduler:** `CosineAnnealingLR(T_max=10, eta_min=1e-6)`
* **Loss Function:** `nn.CrossEntropyLoss()`
* **Batch Size:** 16 (`num_workers=2`, `pin_memory=True`, GPU execution)
* **Hardware Accelerator:** Google Colab NVIDIA Tesla T4 GPU
* **Epochs:** 10

### 4.4 Epoch-by-Epoch Progress Log
During Phase 3 training, model more effectively uses all 10 epochs, suggesting lower likelihood of overfitting having occured:

```text
Epoch 1: Loss = 3.9455 | Training Accuracy = 11.40%
Epoch 2: Loss = 2.1682 | Training Accuracy = 47.72%
Epoch 3: Loss = 1.1416 | Training Accuracy = 72.62%
Epoch 4: Loss = 0.6448 | Training Accuracy = 86.23%
Epoch 5: Loss = 0.4040 | Training Accuracy = 92.56%
Epoch 6: Loss = 0.2764 | Training Accuracy = 96.19%
Epoch 7: Loss = 0.2100 | Training Accuracy = 97.60%
Epoch 8: Loss = 0.1699 | Training Accuracy = 98.59%
Epoch 9: Loss = 0.1438 | Training Accuracy = 98.95%
Epoch 10: Loss = 0.1422 | Training Accuracy = 98.89%
```
### 4.5 Quantitative Validation Results
Validation evaluation was executed across 3,333 unseen images using vectorized tensor lookup maps (`v2f_tensor`, `v2m_tensor`) to measure accuracy across all three benchmark taxonomy levels:

* **Variant Accuracy (100 Classes):** `83.56%`
* **Family Accuracy (70 Classes):** `91.15%`
* **Manufacturer Accuracy (30 Classes):** `95.14%`

### 4.6 Analysis of results
* Model improved minorly by around 3% across all accuracy ratings in validation testing.
* The hierarchical performance remained intact as expected:
$$\text{Variant Accuracy (83.56\%)} < \text{Family Accuracy (91.15\%)} < \text{Manufacturer Accuracy (95.14\%)}$$
   * Backbone understanding of aircraft across all categories has improved slightly.
* Small increase but significant slower training across 10 epochs suggests that with more hyperparameter tuning a greater accuracy is possible.

---

## 5. Phase 4 Further Data Augmentation (`v4.0`)

### 5.1 Objective
Prevent training saturation and augment the training images to encourage deeper understanding of different smaller plane properties (e.g. winglets), increase input resolution to capture differences within an aircraft family. 

### 5.2 Implementation Requirements

1. **Input Resolution Increase:**
   Update sizing from 288x288 &rarr; 384x384

2. **Data Augmentation:**
   Replace manual `ColorJitter` with automated photometric and geomtric transformations to randomise transformations each image undergoes
   ```python
   v2.RandAugment(num_ops=2, magnitude=9)
   ```

### 5.3 Configuration Parameters
* **Backbone:** ConvNeXt V2 Tiny (Pre-trained on ImageNet-1K)
* **Backbone Status:** Unfrozen (`requires_grad = True`)
* **Classifier Head:** Linear layer ($768 \rightarrow 100$)
* **Optimizer:** `AdamW(backbone lr=1e-5, head lr=1e-4, weight_decay=0.05)` (Filtering parameters where `requires_grad=True`)
* **Learning Rate Scheduler:** `CosineAnnealingLR(T_max=10, eta_min=1e-6)`
* **Loss Function:** `nn.CrossEntropyLoss()`
* **Batch Size:** 16 (`num_workers=2`, `pin_memory=True`, GPU execution)
* **Hardware Accelerator:** Google Colab NVIDIA Tesla T4 GPU
* **Epochs:** 10

### 5.4 Epoch-by-Epoch Progress Log
During Phase 3 training, model more effectively uses all 10 epochs, suggesting lower likelihood of overfitting having occured:

```text
Epoch 1: Loss = 3.9498 | Training Accuracy = 10.68%
Epoch 2: Loss = 2.3384 | Training Accuracy = 40.85%
Epoch 3: Loss = 1.3370 | Training Accuracy = 66.23%
Epoch 4: Loss = 0.8213 | Training Accuracy = 80.50%
Epoch 5: Loss = 0.5541 | Training Accuracy = 88.72%
Epoch 6: Loss = 0.3917 | Training Accuracy = 92.95%
Epoch 7: Loss = 0.3047 | Training Accuracy = 95.68%
Epoch 8: Loss = 0.2503 | Training Accuracy = 97.06%
Epoch 9: Loss = 0.2213 | Training Accuracy = 97.39%
Epoch 10: Loss = 0.1943 | Training Accuracy = 97.75%
```
### 5.5 Quantitative Validation Results
Validation evaluation was executed across 3,333 unseen images using vectorized tensor lookup maps (`v2f_tensor`, `v2m_tensor`) to measure accuracy across all three benchmark taxonomy levels:

* **Variant Accuracy (100 Classes):** `86.35%`
* **Family Accuracy (70 Classes):** `92.83%`
* **Manufacturer Accuracy (30 Classes):** `96.25%`

### 5.6 Analysis of results
* Model improved minorly increasing by the greatest % points in the Variant accuracy
* The hierarchical performance remained intact as expected:
$$\text{Variant Accuracy (86.35\%)} < \text{Family Accuracy (92.83\%)} < \text{Manufacturer Accuracy (96.25\%)}$$
   * Greatest increase of the 3 accuracies in variant accuracy
* Greater accuracy acheived through reducing training overfitting
* Sufficient training and hyperparameter analysis for final test

---

## 6. Phase 5 Final Test And Analysis Of Model V4.0

### 6.1 Final Test Results
The final test for this dataset was conducted on 3,333 unseen images. The accuracy of variant, family, and manufacturer classification was calculated as follows.
* **Variant Accuracy (100 Classes):** `87.61%`
* **Family Accuracy (70 Classes):** `93.40%`
* **Manufacturer Accuracy (30 Classes):** `96.34%`