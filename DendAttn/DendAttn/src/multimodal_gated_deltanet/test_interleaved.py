# -*- coding: utf-8 -*-
"""
Test script for interleaved multimodal mode in MultimodalGatedDeltaNet.

Tests:
1. Sequence-level modality_ids (backward compatible)
2. Token-level explicit modality_ids
3. Auto-infer modality_ids from input_ids
4. Update mask generation correctness
5. Gradient flow verification
6. Mixed batch patterns
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multimodal_gated_deltanet.layer import (
    MultimodalGatedDeltaNet,
    MODALITY_TEXT,
    MODALITY_VISION,
    MODALITY_SHARED
)


def test_sequence_level_backward_compatible():
    """Test that sequence-level modality_ids [B] still works."""
    print("\n" + "="*60)
    print("Test 1: Sequence-level modality_ids (backward compatible)")
    print("="*60)

    # Create layer without interleaved mode (default)
    layer = MultimodalGatedDeltaNet(
        hidden_size=256,
        num_heads=4,
        head_dim=32,
        expand_v=2,
    ).cuda()

    # Input
    batch_size, seq_len = 2, 128
    hidden_states = torch.randn(batch_size, seq_len, 256).cuda()

    # Sequence-level modality_ids: [B]
    modality_ids = torch.tensor([0, 1]).cuda()  # batch 0 is text, batch 1 is vision

    # Forward pass
    output, _, _, _ = layer(hidden_states, modality_ids=modality_ids)

    print(f"  Input shape: {hidden_states.shape}")
    print(f"  Modality IDs shape: {modality_ids.shape}")
    print(f"  Output shape: {output.shape}")

    assert output.shape == hidden_states.shape
    print("  ✓ PASSED: Sequence-level modality_ids works correctly")


def test_token_level_explicit():
    """Test token-level modality_ids with explicit input."""
    print("\n" + "="*60)
    print("Test 2: Token-level modality_ids (explicit)")
    print("="*60)

    # Create layer with interleaved mode
    layer = MultimodalGatedDeltaNet(
        hidden_size=256,
        num_heads=4,
        head_dim=32,
        expand_v=2,
        interleaved_mode=True,
    ).cuda()
    layer.eval()  # Use eval mode for short sequences

    # Input - use longer sequence (>64) to ensure chunk mode
    batch_size, seq_len = 2, 128
    hidden_states = torch.randn(batch_size, seq_len, 256).cuda()

    # Token-level modality_ids: [B, L]
    # Sequence: [bos, img, img, img, text, text, text, eos, text...]
    modality_ids = torch.full((batch_size, seq_len), MODALITY_TEXT, dtype=torch.long).cuda()
    # batch 0: bos + images at positions 1-3, eos at 7
    modality_ids[0, 0] = MODALITY_SHARED  # bos
    modality_ids[0, 1:4] = MODALITY_VISION  # images
    modality_ids[0, 7] = MODALITY_SHARED  # eos
    modality_ids[0, -1] = MODALITY_SHARED  # final special token
    # batch 1: bos + text at positions 1-3, images at 4-6, eos at 7
    modality_ids[1, 0] = MODALITY_SHARED
    modality_ids[1, 4:7] = MODALITY_VISION
    modality_ids[1, 7] = MODALITY_SHARED
    modality_ids[1, -1] = MODALITY_SHARED

    # Forward pass
    with torch.no_grad():
        output, _, _, _ = layer(hidden_states, modality_ids=modality_ids)

    print(f"  Input shape: {hidden_states.shape}")
    print(f"  Modality IDs shape: {modality_ids.shape}")
    print(f"  Output shape: {output.shape}")

    assert output.shape == hidden_states.shape
    print("  ✓ PASSED: Token-level modality_ids works correctly")


def test_auto_infer_from_input_ids():
    """Test automatic modality inference from input_ids."""
    print("\n" + "="*60)
    print("Test 3: Auto-infer modality_ids from input_ids")
    print("="*60)

    # Special token IDs
    BOS_TOKEN_ID = 1
    EOS_TOKEN_ID = 2
    PAD_TOKEN_ID = 0
    IMAGE_TOKEN_ID = 32000

    # Create layer with interleaved mode and token IDs
    layer = MultimodalGatedDeltaNet(
        hidden_size=256,
        num_heads=4,
        head_dim=32,
        expand_v=2,
        interleaved_mode=True,
        image_token_id=IMAGE_TOKEN_ID,
        bos_token_id=BOS_TOKEN_ID,
        eos_token_id=EOS_TOKEN_ID,
        pad_token_id=PAD_TOKEN_ID,
    ).cuda()
    layer.eval()  # Use eval mode for testing

    # Input - use longer sequence (>64) to ensure chunk mode
    batch_size, seq_len = 2, 128
    hidden_states = torch.randn(batch_size, seq_len, 256).cuda()

    # Simulated input_ids: [bos, img, img, img, text, text, text, eos, pad, ...]
    input_ids = torch.full((batch_size, seq_len), 100, dtype=torch.long).cuda()  # Default text tokens
    # Batch 0: bos, images, text, eos, padding
    input_ids[0, 0] = BOS_TOKEN_ID
    input_ids[0, 1:4] = IMAGE_TOKEN_ID
    input_ids[0, 7] = EOS_TOKEN_ID
    input_ids[0, 8:] = PAD_TOKEN_ID
    # Batch 1: bos, text, images, eos, padding
    input_ids[1, 0] = BOS_TOKEN_ID
    input_ids[1, 4:7] = IMAGE_TOKEN_ID
    input_ids[1, 7] = EOS_TOKEN_ID
    input_ids[1, 8:] = PAD_TOKEN_ID

    # Forward pass - modality_ids should be inferred automatically
    with torch.no_grad():
        output, _, _, _ = layer(hidden_states, input_ids=input_ids)

    print(f"  Input shape: {hidden_states.shape}")
    print(f"  Input IDs shape: {input_ids.shape}")
    print(f"  Output shape: {output.shape}")

    # Verify inference
    inferred_modality_ids = layer._infer_modality_ids(
        input_ids, batch_size, seq_len, input_ids.device
    )
    print(f"  Inferred modality_ids (first 16 positions):")
    print(f"    Batch 0: {inferred_modality_ids[0, :16].tolist()}")
    print(f"    Batch 1: {inferred_modality_ids[1, :16].tolist()}")

    # Check expected values for first 16 positions
    expected_batch0_first16 = torch.tensor([-1, 1, 1, 1, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1])
    expected_batch1_first16 = torch.tensor([-1, 0, 0, 0, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1])

    assert torch.equal(inferred_modality_ids[0, :16].cpu(), expected_batch0_first16), \
        f"Batch 0 mismatch: {inferred_modality_ids[0, :16].cpu()} vs {expected_batch0_first16}"
    assert torch.equal(inferred_modality_ids[1, :16].cpu(), expected_batch1_first16), \
        f"Batch 1 mismatch: {inferred_modality_ids[1, :16].cpu()} vs {expected_batch1_first16}"

    print("  ✓ PASSED: Auto-inference from input_ids works correctly")


def test_update_mask_token_level():
    """Test that update mask is correctly generated for token-level modality."""
    print("\n" + "="*60)
    print("Test 4: Update mask generation for token-level modality")
    print("="*60)

    layer = MultimodalGatedDeltaNet(
        hidden_size=256,
        num_heads=4,
        head_dim=32,
        expand_v=2,
        interleaved_mode=True,
    ).cuda()

    batch_size, seq_len = 1, 128
    device = torch.device('cuda')
    dtype = torch.float32

    # Token-level modality_ids (first 8 positions meaningful, rest is text)
    # Position: 0    1    2    3    4    5    6    7    ...
    # Modality: -1   1    1    -1   0    0    0    -1   0 (rest)
    modality_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    modality_ids[0, 0] = MODALITY_SHARED
    modality_ids[0, 1:3] = MODALITY_VISION
    modality_ids[0, 3] = MODALITY_SHARED
    modality_ids[0, 4:7] = MODALITY_TEXT  # already 0
    modality_ids[0, 7] = MODALITY_SHARED

    update_mask = layer._get_update_mask(
        modality_ids, batch_size, seq_len, device, dtype
    )

    print(f"  Modality IDs (first 8): {modality_ids[0, :8].tolist()}")
    print(f"  Update mask shape: {update_mask.shape}")  # [E=3, B=1, L=128, H=4]

    # Check Expert 0 (Shared): all 1s
    assert torch.all(update_mask[0] == 1.0), "Expert 0 should always be 1"
    print("  ✓ Expert 0 (Shared): All positions active")

    # Check Expert 1 (Text): positions 4,5,6 and 8+ active (modality_id == 0)
    expert1_mask = update_mask[1, 0, :8, 0].cpu()  # [L] first 8 positions
    expected_expert1 = torch.tensor([0, 0, 0, 0, 1, 1, 1, 0], dtype=torch.float32)
    assert torch.equal(expert1_mask, expected_expert1), \
        f"Expert 1 mismatch: {expert1_mask} vs {expected_expert1}"
    print(f"  ✓ Expert 1 (Text) first 8: {expert1_mask.tolist()}")

    # Check Expert 2 (Vision): only positions 1,2 active (modality_id == 1)
    expert2_mask = update_mask[2, 0, :8, 0].cpu()  # [L] first 8 positions
    expected_expert2 = torch.tensor([0, 1, 1, 0, 0, 0, 0, 0], dtype=torch.float32)
    assert torch.equal(expert2_mask, expected_expert2), \
        f"Expert 2 mismatch: {expert2_mask} vs {expected_expert2}"
    print(f"  ✓ Expert 2 (Vision) first 8: {expert2_mask.tolist()}")

    print("  ✓ PASSED: Update mask generation is correct")


def test_gradient_flow():
    """Test that gradients flow correctly in interleaved mode."""
    print("\n" + "="*60)
    print("Test 5: Gradient flow in interleaved mode")
    print("="*60)

    layer = MultimodalGatedDeltaNet(
        hidden_size=256,
        num_heads=4,
        head_dim=32,
        expand_v=2,
        interleaved_mode=True,
    ).cuda()

    batch_size, seq_len = 2, 128
    # Create on cuda directly with requires_grad
    hidden_states = torch.randn(batch_size, seq_len, 256, device='cuda', requires_grad=True)

    # Token-level modality_ids with mixed modalities
    modality_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device='cuda')
    modality_ids[:, :10] = MODALITY_SHARED  # bos, special tokens
    modality_ids[:, 10:40] = MODALITY_VISION  # image patches
    modality_ids[:, 40:] = MODALITY_TEXT  # text tokens

    # Forward pass
    output, _, _, _ = layer(hidden_states, modality_ids=modality_ids)

    # Backward pass
    loss = output.sum()
    loss.backward()

    # Check gradients
    params_with_grad = sum(1 for p in layer.parameters() if p.grad is not None)
    total_params = sum(1 for p in layer.parameters())

    print(f"  Parameters with gradients: {params_with_grad}/{total_params}")
    print(f"  Input gradient exists: {hidden_states.grad is not None}")
    if hidden_states.grad is not None:
        print(f"  Input gradient norm: {hidden_states.grad.norm().item():.4f}")

    assert hidden_states.grad is not None
    assert hidden_states.grad.norm() > 0
    print("  ✓ PASSED: Gradients flow correctly")


def test_mixed_batch():
    """Test with different modality patterns in same batch."""
    print("\n" + "="*60)
    print("Test 6: Mixed modality patterns in batch")
    print("="*60)

    IMAGE_TOKEN_ID = 32000

    layer = MultimodalGatedDeltaNet(
        hidden_size=256,
        num_heads=4,
        head_dim=32,
        expand_v=2,
        interleaved_mode=True,
        image_token_id=IMAGE_TOKEN_ID,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    ).cuda()
    layer.eval()  # Use eval mode for testing

    batch_size, seq_len = 4, 128
    hidden_states = torch.randn(batch_size, seq_len, 256).cuda()

    # Different patterns in each batch item
    input_ids = torch.full((batch_size, seq_len), 100).cuda()  # Default text

    # Batch 0: Image first, then text
    input_ids[0, 0] = 1  # bos
    input_ids[0, 1:9] = IMAGE_TOKEN_ID
    input_ids[0, -1] = 2  # eos

    # Batch 1: Text first, then image
    input_ids[1, 0] = 1
    input_ids[1, 60:68] = IMAGE_TOKEN_ID
    input_ids[1, -1] = 2

    # Batch 2: Multiple image segments
    input_ids[2, 0] = 1
    input_ids[2, 10:20] = IMAGE_TOKEN_ID
    input_ids[2, 80:90] = IMAGE_TOKEN_ID
    input_ids[2, -1] = 2

    # Batch 3: Pure text
    input_ids[3, 0] = 1
    input_ids[3, -1] = 2

    # Forward pass
    with torch.no_grad():
        output, _, _, _ = layer(hidden_states, input_ids=input_ids)

    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Output shape: {output.shape}")

    # Check output is valid
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

    print("  ✓ PASSED: Mixed batch patterns work correctly")


def main():
    print("\n" + "="*60)
    print("  INTERLEAVED MULTIMODAL MODE TESTS")
    print("="*60)

    test_sequence_level_backward_compatible()
    test_token_level_explicit()
    test_auto_infer_from_input_ids()
    test_update_mask_token_level()
    test_gradient_flow()
    test_mixed_batch()

    print("\n" + "="*60)
    print("  ALL 6 TESTS PASSED!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
