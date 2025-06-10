# PyTorch Event Stream Prediction Model

## Overview
This repository contains a transformer-based event stream prediction model for the PyTorch Model Challenge. The model processes multiple event streams to predict future events by learning entity state representations.

## Current Status & Performance

### Training Progress
- **Current Loss**: ~85 (GPU with bfloat16)
- **Target Loss**: ~4 (based on CPU baseline)
- **Status**: Model training successfully, loss tracking working
- **Next Steps**: Optimize for better convergence

### Known Issues & Solutions
1. **High GPU Loss vs CPU**: Currently investigating bfloat16 vs float32 precision
2. **Loss Tracking Fixed**: Category averages now display correctly
3. **Backward Pass Added**: Model now properly updates weights

## Architecture Highlights

* **Multiple transformer encoders** for different event types
* **Cross-attention mechanism** between encoded states and target sequences  
* **Event-specific embeddings** with positional encoding
* **GPU-optimized training** with bfloat16 precision
* **Comprehensive loss tracking** per category

## Current Configuration

### Model Settings
```json
{
  "src_seq_len": 200,
  "tgt_seq_len": 300,
  "id_category_size": 1000,
  "epochs": 500,
  "batch_size": 8,
  "grad_accum": 8,
  "model": {
    "d_model": 64,
    "num_heads": 8,
    "encoder_layers": 3,
    "decoder_layers": 5
  }
}