# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rank detection utilities."""

from __future__ import annotations

import logging

import torch

from . import p2p_pb2

logger = logging.getLogger("modelexpress.rank_utils")


def get_global_rank(device: torch.device) -> int:
    """Get the global distributed rank for this worker."""
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            rank = dist.get_rank()
            logger.debug(f"Got global rank from torch.distributed: {rank}")
            return rank
    except (ImportError, RuntimeError) as e:
        logger.debug(f"Could not get global rank from torch.distributed: {e}")

    if hasattr(device, "index") and device.index is not None:
        logger.debug(f"Using device.index as global rank fallback: {device.index}")
        return device.index

    return 0


def compute_draft_slot(draft_idx: int | None) -> int:
    """
    Compute the draft slot index for a given draft model index.
    Args:
        draft_idx: The index of the draft model. If None, it represents the main model.

    Returns:
        The computed draft slot index. Returns 0 for the main model (None),
        otherwise returns draft_idx + 1. (1,2,3,...)

    """
    return 0 if draft_idx is None else draft_idx + 1 # if None, then it is main model with index 0. Otherwise, idx + 1


def compute_port(base_port: int, device_id: int, draft_idx: int | None, max_draft_models: int) -> int:
    """
    Compute the port number for a given device ID and draft model index.
    """
    return base_port + device_id * (max_draft_models + 1) + compute_draft_slot(draft_idx)


def get_draft_model_idx(identity: p2p_pb2.SourceIdentity) -> int | None:
    """
    Get the draft model index from the source identity.
    Returns:
        The draft model index if it exists, otherwise None.
    """
    return identity.draft_model_idx if identity.HasField("draft_model_idx") else None
