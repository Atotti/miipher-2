# Miipher-2 Accelerate Migration Guide

## Overview
This guide documents the migration of Miipher-2 training scripts from manual PyTorch CUDA management to Hugging Face Accelerate framework. All existing training files have been updated to use Accelerate for improved multi-GPU support, automatic mixed precision, and simplified distributed training.

## Updated Files

### Core Training Files
All training implementations have been updated to use Accelerate:

1. **`src/miipher_2/train/adapter.py`** - Adapter training (✅ Complete)
   - Full Accelerate integration with distributed validation
   - Automatic mixed precision support
   - Simplified checkpoint management

2. **`src/miipher_2/train/hifigan.py`** - HiFi-GAN fine-tuning (✅ Complete)
   - Generator and Discriminator training with Accelerate
   - Multi-GPU loss synchronization
   - Automatic gradient accumulation

3. **`src/miipher_2/train/pre_train_vocoder.py`** - HiFi-GAN pre-training (✅ Complete)
   - Clean audio pre-training with Accelerate
   - Distributed validation with loss gathering
   - Simplified checkpoint handling

### Utility Files
- **`src/miipher_2/accelerate_utils.py`** - New shared utilities
  - `build_accelerator()` - Standardized Accelerator setup
  - `log_metrics()` - WandB logging with main process guard
  - `print_main()` - Main process-only printing

### Configuration Files
Updated config files with Accelerate settings:

- **`configs/adapter_layer_6.yaml`** - Adapter training config
- **`configs/hifigan_pretrain_layer_6.yaml`** - HiFi-GAN pre-training config
- **`configs/hifigan_finetune_layer_6.yaml`** - HiFi-GAN fine-tuning config

## Key Changes Made

### 1. Device Management
**Before:**
```python
model = Model().cuda()
data = data.cuda()
```

**After:**
```python
model = Model()
model, data_loader = accelerator.prepare(model, data_loader)
# Device management handled automatically
```

### 2. Training Loops
**Before:**
```python
loss.backward()
if (step + 1) % accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()
```

**After:**
```python
with accelerator.accumulate(model):
    loss = compute_loss()
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
```

### 3. Validation with Multi-GPU
**Before:**
```python
# Manual loss aggregation
loss_sum += loss.item()
```

**After:**
```python
# Automatic gathering across GPUs
loss_gathered = accelerator.gather(loss.unsqueeze(0))
if accelerator.is_main_process:
    avg_loss = loss_gathered.mean().item()
```

### 4. Checkpointing
**Before:**
```python
torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'step': step
}, path)
```

**After:**
```python
accelerator.save_state(output_dir=save_dir)
# Handles model, optimizer, scheduler, and RNG states automatically
```

## Usage

### Running Training
Use `accelerate launch` prefix for all training commands:

```bash
# Adapter training
accelerate launch cmd/train_adapter.py --config-name adapter_layer_6

# HiFi-GAN pre-training
accelerate launch cmd/pre_train_vocoder.py --config-name hifigan_pretrain_layer_6

# HiFi-GAN fine-tuning
accelerate launch cmd/finetune_vocoder.py --config-name hifigan_finetune_layer_6
```

### Multi-GPU Training
```bash
# Using all available GPUs
accelerate launch --multi_gpu cmd/train_adapter.py --config-name adapter_layer_6

# Using specific GPUs
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu cmd/train_adapter.py --config-name adapter_layer_6
```

### Configuration Options
Each config file now includes Accelerate settings:

```yaml
# Training settings
training:
  mixed_precision: "fp16"  # or "bf16", "no"
  force_cpu: false
  gradient_accumulation_steps: 4

# Advanced settings
accelerate:
  fsdp:
    enabled: false  # Enable for very large models
  deepspeed:
    enabled: false  # Enable for advanced optimizations
```

## Benefits

1. **Simplified Multi-GPU Support**: Automatic data parallel training
2. **Mixed Precision**: Faster training with reduced memory usage
3. **Better Checkpointing**: Automatic handling of all training states
4. **Distributed Training**: Easy scaling to multiple nodes
5. **Memory Efficiency**: Optimized memory usage patterns
6. **Maintainability**: Cleaner, more standardized code

## Migration Summary

✅ **Completed Tasks:**
- Updated all 3 core training files to use Accelerate
- Created shared utility functions in `accelerate_utils.py`
- Updated configuration files with Accelerate settings
- Maintained backward compatibility with existing command structure
- Updated validation functions for distributed training
- Implemented proper gradient accumulation and mixed precision

🔧 **Technical Improvements:**
- Removed manual CUDA device management
- Simplified checkpoint save/load logic
- Added distributed validation with loss gathering
- Implemented main process guards for logging
- Added automatic gradient clipping integration

## Troubleshooting

### Common Issues
1. **Out of Memory**: Reduce batch size or enable gradient checkpointing
2. **Slow Training**: Check if mixed precision is enabled
3. **Checkpoint Loading**: Ensure checkpoint directory structure is correct

### Example Commands
```bash
# Check accelerate configuration
accelerate config

# Debug mode with single GPU
accelerate launch --num_processes=1 cmd/train_adapter.py --config-name adapter_layer_6

# Monitor training
accelerate launch cmd/train_adapter.py --config-name adapter_layer_6 2>&1 | tee training.log
```

The migration is now complete and all training scripts are ready for production use with Accelerate!
