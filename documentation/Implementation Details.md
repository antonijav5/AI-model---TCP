## Implementation Details

### Key Components
1. **EventEncoder**: Handles multi-category event embedding with learnable representations
2. **Entity-Specific Encoders**: Separate transformer encoders for each event stream type
3. **EventPredictor**: Transformer decoder with cross-attention to encoded states
4. **Training Pipeline**: Supports gradient accumulation, checkpointing, and TensorBoard logging

## Technical Decisions

### Model-Level
- **Only last encoder vectors**: Implemented as required, creating an efficient information bottleneck and preserving temporal causality
- **Separate encoders**: Allows specialized learning for each entity type (e.g., Actor, Location, Time), supporting modularity and stream-specific context
- **Mish activation**: Chosen for smoother gradients and better convergence properties compared to ReLU
- **Cross-attention between streams**: Enables contextual fusion across different encoded modalities
- **Multi-loss per category**: Allows fine-grained feedback and better understanding of learning dynamics

### Optimization & Training
- **Gradient Clipping (`max_norm=1.0`)**: Prevents exploding gradients and stabilizes training across all steps
- **Learning Rate Scheduler (`ReduceLROnPlateau`)**: Dynamically lowers learning rate when validation loss plateaus — boosting convergence without overfitting
- **No multiprocessing in DataLoader**: Simplifies environment setup and reduces CPU thread contention when running in containerized/Docker environments
- **Precision control**: Set `torch.set_default_dtype(torch.float32)` and `set_float32_matmul_precision('high')` to stabilize computations during training

### Infrastructure
- **Checkpointing system**: Saves best model and latest model separately, ensuring safe recovery and reproducibility
- **TensorBoard logging**: Enables detailed tracking of total loss, per-category loss, and timing stats for both training and validation

## Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Memory constraints with workers | Set `num_workers=0`, disabled persistent workers |
| Docker container space issues | Optimized batch processing, shared volumes |
| Dimension mismatches | Careful tensor reshaping in encoder aggregation |
| CPU training speed | Increased batch size, optimized data loading |

## Conclusions

The implementation successfully demonstrates:
1. **Understanding of transformer architectures** applied to event sequences
2. **Ability to handle complex multi-stream inputs** with proper attention mechanisms
3. **Practical problem-solving** in adapting the model for local CPU training
4. **Clean, modular code** that follows PyTorch best practices

The model learns to predict future events by maintaining entity states through their event histories, exactly as intended by the challenge design.