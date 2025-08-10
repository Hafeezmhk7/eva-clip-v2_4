"""
Distributed Communication Utilities for BLIP3-o
src/modules/distributed/communication.py

Provides communication and synchronization utilities for distributed training.
"""

import torch
import torch.distributed as dist
import logging
from typing import Dict, Any, Optional, List
import json

logger = logging.getLogger(__name__)


class DistributedCommunicator:
    """
    Handles distributed communication and synchronization for BLIP3-o training
    """
    
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.is_distributed = dist.is_initialized() if hasattr(dist, 'is_initialized') else False
        
    def barrier(self, timeout_seconds: Optional[int] = None):
        """Synchronize all processes"""
        if self.is_distributed:
            try:
                if timeout_seconds:
                    dist.barrier(timeout=torch.distributed.default_pg_timeout)
                else:
                    dist.barrier()
            except Exception as e:
                logger.warning(f"Barrier failed on rank {self.rank}: {e}")
    
    def is_master(self) -> bool:
        """Check if current process is master (rank 0)"""
        return self.rank == 0
    
    def broadcast_object(self, obj: Any, src_rank: int = 0) -> Any:
        """Broadcast an object from source rank to all ranks"""
        if not self.is_distributed:
            return obj
        
        try:
            object_list = [obj] if self.rank == src_rank else [None]
            dist.broadcast_object_list(object_list, src=src_rank)
            return object_list[0]
        except Exception as e:
            logger.warning(f"Broadcast failed on rank {self.rank}: {e}")
            return obj
    
    def all_gather_object(self, obj: Any) -> List[Any]:
        """Gather objects from all ranks"""
        if not self.is_distributed:
            return [obj]
        
        try:
            gathered_objects = [None] * self.world_size
            dist.all_gather_object(gathered_objects, obj)
            return gathered_objects
        except Exception as e:
            logger.warning(f"All-gather failed on rank {self.rank}: {e}")
            return [obj]
    
    def reduce_tensor(self, tensor: torch.Tensor, op=dist.ReduceOp.SUM, dst: int = 0) -> torch.Tensor:
        """Reduce tensor across all ranks"""
        if not self.is_distributed:
            return tensor
        
        try:
            dist.reduce(tensor, dst=dst, op=op)
            return tensor
        except Exception as e:
            logger.warning(f"Reduce failed on rank {self.rank}: {e}")
            return tensor
    
    def all_reduce_tensor(self, tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
        """All-reduce tensor across all ranks"""
        if not self.is_distributed:
            return tensor
        
        try:
            dist.all_reduce(tensor, op=op)
            return tensor
        except Exception as e:
            logger.warning(f"All-reduce failed on rank {self.rank}: {e}")
            return tensor


class MetricsAggregator:
    """
    Aggregates metrics across distributed training ranks
    """
    
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.communicator = DistributedCommunicator(rank, world_size)
    
    def aggregate_metrics(self, local_metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Aggregate metrics from all ranks
        
        Args:
            local_metrics: Metrics from current rank
            
        Returns:
            Aggregated metrics (only on rank 0, None on other ranks)
        """
        if not self.communicator.is_distributed:
            return local_metrics
        
        try:
            # Gather metrics from all ranks
            all_metrics = self.communicator.all_gather_object(local_metrics)
            
            if self.rank == 0:
                # Aggregate metrics on master rank
                aggregated = {}
                
                # Collect all metric keys
                all_keys = set()
                for metrics in all_metrics:
                    if isinstance(metrics, dict):
                        all_keys.update(metrics.keys())
                
                # Aggregate each metric
                for key in all_keys:
                    values = []
                    total_weight = 0
                    
                    for rank_idx, metrics in enumerate(all_metrics):
                        if isinstance(metrics, dict) and key in metrics:
                            value = metrics[key]
                            
                            # Handle different value types
                            if isinstance(value, (int, float)) and not isinstance(value, bool):
                                # Weight by rank contribution (simple equal weighting)
                                weight = 1.0
                                values.append((value, weight))
                                total_weight += weight
                            elif isinstance(value, str):
                                # For string values, just take the first valid one
                                if key not in aggregated:
                                    aggregated[key] = value
                            elif isinstance(value, bool):
                                # For boolean values, take logical OR
                                aggregated[key] = aggregated.get(key, False) or value
                    
                    # Compute weighted average for numeric values
                    if values and total_weight > 0:
                        weighted_sum = sum(v * w for v, w in values)
                        aggregated[key] = weighted_sum / total_weight
                
                # Add aggregation metadata
                aggregated['_aggregation_info'] = {
                    'num_ranks': len(all_metrics),
                    'aggregated_by_rank': 0,
                    'aggregation_method': 'weighted_average'
                }
                
                return aggregated
            else:
                # Non-master ranks return None
                return None
                
        except Exception as e:
            logger.warning(f"Metrics aggregation failed on rank {self.rank}: {e}")
            if self.rank == 0:
                return local_metrics
            else:
                return None
    
    def log_aggregated_metrics(self, metrics: Dict[str, Any], prefix: str = ""):
        """Log aggregated metrics (only on rank 0)"""
        if self.rank != 0 or not metrics:
            return
        
        logger.info(f"📊 {prefix}Aggregated metrics across {self.world_size} ranks:")
        for key, value in metrics.items():
            if not key.startswith('_') and isinstance(value, (int, float)):
                logger.info(f"   {key}: {value:.6f}")


def log_distributed_info(rank: int, world_size: int, message: str, level: str = "info"):
    """
    Log distributed training information
    
    Args:
        rank: Current process rank
        world_size: Total number of processes
        message: Message to log
        level: Log level ('info', 'warning', 'error')
    """
    
    if rank == 0:  # Only master rank logs by default
        log_msg = f"[Distributed {world_size}GPUs] {message}"
        
        if level == "info":
            logger.info(log_msg)
        elif level == "warning":
            logger.warning(log_msg)
        elif level == "error":
            logger.error(log_msg)
        else:
            logger.info(log_msg)


def sync_random_seed(seed: int, rank: int, world_size: int) -> int:
    """
    Synchronize random seed across all ranks
    
    Args:
        seed: Base seed value
        rank: Current process rank  
        world_size: Total number of processes
        
    Returns:
        Synchronized seed for this rank
    """
    
    # Create rank-specific seed while maintaining reproducibility
    rank_seed = seed + rank * 10000
    
    if rank == 0:
        logger.info(f"🎲 Synchronized random seeds across {world_size} ranks (base seed: {seed})")
    
    return rank_seed


def wait_for_everyone(rank: int, world_size: int, message: str = "Synchronizing"):
    """
    Wait for all ranks to reach this point
    
    Args:
        rank: Current process rank
        world_size: Total number of processes
        message: Description of what we're waiting for
    """
    
    if rank == 0:
        logger.info(f"⏳ {message} - waiting for all {world_size} ranks...")
    
    if dist.is_initialized():
        try:
            dist.barrier()
            if rank == 0:
                logger.info(f"✅ All {world_size} ranks synchronized")
        except Exception as e:
            logger.warning(f"Synchronization failed on rank {rank}: {e}")


def check_distributed_setup(rank: int, world_size: int) -> Dict[str, Any]:
    """
    Check distributed training setup
    
    Args:
        rank: Current process rank
        world_size: Total number of processes
        
    Returns:
        Dictionary with setup information
    """
    
    setup_info = {
        'rank': rank,
        'world_size': world_size,
        'is_distributed': dist.is_initialized() if hasattr(dist, 'is_initialized') else False,
        'backend': None,
        'master_addr': None,
        'master_port': None,
    }
    
    if setup_info['is_distributed']:
        try:
            setup_info['backend'] = dist.get_backend()
            setup_info['master_addr'] = os.environ.get('MASTER_ADDR', 'unknown')
            setup_info['master_port'] = os.environ.get('MASTER_PORT', 'unknown')
        except Exception as e:
            logger.warning(f"Could not get distributed setup info: {e}")
    
    if rank == 0:
        logger.info("🔧 Distributed setup check:")
        logger.info(f"   Rank: {rank}/{world_size}")
        logger.info(f"   Distributed: {setup_info['is_distributed']}")
        if setup_info['is_distributed']:
            logger.info(f"   Backend: {setup_info['backend']}")
            logger.info(f"   Master: {setup_info['master_addr']}:{setup_info['master_port']}")
    
    return setup_info


# Compatibility import
import os