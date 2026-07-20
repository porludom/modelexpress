# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rank detection utilities."""

from __future__ import annotations

import re
import logging

import torch

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

def parse_draft_model_idx(model_name: str) -> int | None:
    """Extract draft_model_idx from model name."""

    match = re.search(r"::draft(\d+)$", model_name)
    if match:
        return int(match.group(1))
    return None

def compute_draft_slot(draft_idx: int | None) -> int:
    return 0 if draft_idx is None else draft_idx + 1 # if None, then it is main model with index 0. Otherwise, idx + 1