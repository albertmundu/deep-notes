# Model Compression

Important papers related to model compression that can help in model deployment for edge devices.

## 1. Papers
1. Up or Down? Adaptive Rounding for Post-Training Quantization [[paper](https://arxiv.org/abs/2004.10568)]


## 2. Webinars
1. AIMET [[video](https://www.youtube.com/watch?v=1Q3I4OKU29I)]


## 3. AI Model Efficiency Toolkit (AIMET)
### Benefits of AIMET
1. Lower battery power
2. Lower storage
3. Lower memory bandwidth
4. Higher performance
5. Maintains model accuracy
6. Simple ease of use

### Quantization
- state-of-the-art INT8 and INT4 performance
- Post-training quantization methods, including Data-Free Quantization and Adaptive Rounding (AdaRound)
- Quantization-aware training
- Quantization simulation

### Compresion 
- Efficient tensor decomposition and removal of redundant channels in convolutional layers
- Spatial singular value decomposition (SVD)
- Channel pruning


### Visualization
- Analysis tools for drawing insights for quantization and compression 
- Weight ranges
- Per-layer compression sensitivity

### AIMET Model Zoo - Accurate pre-trained 8-bit quantized models
- Image classification
- Object detection
- Semantic segmentation
- Pose estimation
- Super resolution
- Speech recognition

*AIMET Techniques maintain accuracy with the original implementation of the model.*

### Flow

**Cross-layer equalization** (equalize weight ranges) --> **Bias absorption** (avoid high activation ranges) --> **Weight quantization** --> **Bias correction** (measure and correct shift in layer outputs) --> **Activation range estimation** (estimate activation ranges for quantization)

### Example: Simple APIs

```
model = models.resnet18(pretrained=True)

# cross-layer equalization & high-bias absorption
equalize_model(model,input_shape=(1,3,224,224)) 


sim = QuantizationSimModel(model)
sim.compute_encodings(forward_pass_callback=evaluate_model, forward_pass_callback_args=500)

# user pipeline will provide quantized accuracy
evaluate_model(sim.model) 
```

