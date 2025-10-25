# VIT - CIFAR10

## File Structure

| File | Description                                                                           |
|------|---------------------------------------------------------------------------------------|
| `q1.ipynb` | CIFAR-10 ViT implementation, training, evaluation, and attention visualization. |
| `README.md` | This file.                                                                     |
________________________________________________________________________________________________

## Config ( also in the notebook )
```
ViTconfig = {
    "image_size": 32,
    "patch_size": 4,
    "num_classes": 10,
    "num_channels": 3,
    "patchify_method": "conv",
    "hidden_size": 256,
    "num_hidden_layers": 8,
    "num_attention_heads": 8,
    "intermediate_size": 512,
    "hidden_dropout_prob": 0.2,
    "bias": True
}
```
## Evaluation Metrics
* Val Accuracy: 76.12%
* Train Accuracy: 85.06%

## Training Graph with the above given config
* The model was trained on a 1e-3 learning rate, with AdamW optimizer, ReduceonPlateau scheduler
<img width="700" height="470" alt="image" src="https://github.com/user-attachments/assets/18cf3fde-71f0-479f-8e23-bdf4afdd18b2" />


## Visualisation using Rollout Attention

<img src="https://github.com/user-attachments/assets/2a6be0a0-f5ae-4595-80ea-aa2078fe8e8d" style="width:600px; height:auto;" />
<img src="https://github.com/user-attachments/assets/c8b6eb73-45c3-4352-980b-9f320ee75ef1" style="width:600px; height:auto;" />
<img src="https://github.com/user-attachments/assets/490dbe4b-e9ce-450c-b7ab-450059bf8aea" style="width:600px; height:auto;" />
<img src="https://github.com/user-attachments/assets/dde5086f-3bfe-4751-8ddf-fdf82115e131" style="width:600px; height:auto;" />
<img src="https://github.com/user-attachments/assets/5dd36929-1d91-47b2-87e2-e3ee6e4f1457" style="width:600px; height:auto;" />
<img src="https://github.com/user-attachments/assets/01489ad3-8a34-47be-a688-298a081f70ad" style="width:600px; height:auto;" />
<img src="https://github.com/user-attachments/assets/659d21dc-cdc8-4fa0-bd02-4cf9d67db37c" style="width:600px; height:auto;" />


## Aditional comments
* Including Dropout at various places improved accuracy
* Patch Size 4 showed better accuracy from Patch Size 8
* Increasing Num Layers to 8 increased accuracy
* The val loss saturated after ~75th epoch


### Conclusion
A VIT Trained on CIFAR10 is not expected to perform really well given the small size of the dataset, a CNN would have performed better due to its inherent bias which transformer lacks, the paper mentioning the results on CIFAR10 are from a pretrained ViT.

