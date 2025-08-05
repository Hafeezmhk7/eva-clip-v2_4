"""
Inter-GPU Communication Utilities for BLIP3-o Distributed Training
src/modules/distributed/communication.py

Provides utilities for communication between GPUs during distributed training
and multi-GPU extraction processes.
"""

import torch
import torch.distributed as dist
import pickle
import json
import time
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DistributedCommunicator:
    """Handles communication between distributed processes"""
    
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.is_initialized = dist.is_initialized()
        
    def barrier(self, timeout_seconds: int = 300):
        """Synchronize all processes"""
        if self.is_initialized:
            dist.barrier(timeout=timeout_seconds)
    
    def broadcast_object(self, obj: Any, src_rank: int = 0) -> Any:
        """Broadcast an object from src_rank to all other ranks"""
        if not self.is_initialized:
            return obj
            
        if self.rank == src_rank:
            # Serialize object
            obj_bytes = pickle.dumps(obj)
            size_tensor = torch.tensor([len(obj_bytes)], dtype=torch.long)
            
            # Broadcast size first
            dist.broadcast(size_tensor, src=src_rank)
            
            # Then broadcast the object
            obj_tensor = torch.frombuffer(obj_bytes, dtype=torch.uint8)
            dist.broadcast(obj_tensor, src=src_rank)
            
            return obj
        else:
            # Receive size first
            size_tensor = torch.tensor([0], dtype=torch.long)
            dist.broadcast(size_tensor, src=src_rank)
            
            # Receive object
            obj_tensor = torch.zeros(size_tensor.item(), dtype=torch.uint8)
            dist.broadcast(obj_tensor, src=src_rank)
            
            # Deserialize
            obj_bytes = obj_tensor.numpy().tobytes()
            obj = pickle.loads(obj_bytes)
            return obj
    
    def all_gather_object(self, obj: Any) -> List[Any]:
        """Gather objects from all ranks"""
        if not self.is_initialized:
            return [obj]
        
        # Convert object to tensor
        obj_bytes = pickle.dumps(obj)
        local_size = torch.tensor([len(obj_bytes)], dtype=torch.long)
        
        # All gather sizes
        size_list = [torch.tensor([0], dtype=torch.long) for _ in range(self.world_size)]
        dist.all_gather(size_list, local_size)
        
        # Prepare tensors for all gather
        max_size = max([size.item() for size in size_list])
        
        # Pad local tensor to max size
        local_tensor = torch.zeros(max_size, dtype=torch.uint8)
        local_tensor[:len(obj_bytes)] = torch.frombuffer(obj_bytes, dtype=torch.uint8)
        
        # All gather tensors
        tensor_list = [torch.zeros(max_size, dtype=torch.uint8) for _ in range(self.world_size)]
        dist.all_gather(tensor_list, local_tensor)
        
        # Deserialize objects
        obj_list = []
        for i, tensor in enumerate(tensor_list):
            actual_size = size_list[i].item()
            obj_bytes = tensor[:actual_size].numpy().tobytes()
            obj = pickle.loads(obj_bytes)
            obj_list.append(obj)
        
        return obj_list
    
    def reduce_dict(self, input_dict: Dict[str, float], dst_rank: int = 0, 
                    op: str = 'sum') -> Optional[Dict[str, float]]:
        """Reduce dictionary of metrics across all ranks"""
        if not self.is_initialized:
            return input_dict
        
        # Convert to tensors
        keys = list(input_dict.keys())
        values = torch.tensor([input_dict[k] for k in keys], dtype=torch.float32)
        
        # Reduce operation
        reduce_op = getattr(dist.ReduceOp, op.upper())
        dist.reduce(values, dst=dst_rank, op=reduce_op)
        
        if self.rank == dst_rank:
            return {k: v.item() for k, v in zip(keys, values)}
        else:
            return None
    
    def all_reduce_dict(self, input_dict: Dict[str, float], op: str = 'sum') -> Dict[str, float]:
        """All-reduce dictionary of metrics across all ranks"""
        if not self.is_initialized:
            return input_dict
        
        # Convert to tensors
        keys = list(input_dict.keys())
        values = torch.tensor([input_dict[k] for k in keys], dtype=torch.float32)
        
        # All reduce operation
        reduce_op = getattr(dist.ReduceOp, op.upper())
        dist.all_reduce(values, op=reduce_op)
        
        return {k: v.item() for k, v in zip(keys, values)}


class ProgressTracker:
    """Tracks progress across distributed processes"""
    
    def __init__(self, rank: int, world_size: int, total_items: int):
        self.rank = rank
        self.world_size = world_size
        self.total_items = total_items
        self.local_progress = 0
        self.global_progress = 0
        self.start_time = time.time()
        self.communicator = DistributedCommunicator(rank, world_size)
        
    def update(self, increment: int = 1):
        """Update local progress"""
        self.local_progress += increment
        
    def sync_progress(self) -> Dict[str, Any]:
        """Synchronize progress across all ranks"""
        try:
            # Gather progress from all ranks
            progress_data = {
                'rank': self.rank,
                'local_progress': self.local_progress,
                'timestamp': time.time()
            }
            
            all_progress = self.communicator.all_gather_object(progress_data)
            
            # Calculate global progress
            total_progress = sum([p['local_progress'] for p in all_progress])
            self.global_progress = total_progress
            
            # Calculate statistics
            elapsed_time = time.time() - self.start_time
            items_per_second = total_progress / elapsed_time if elapsed_time > 0 else 0
            eta_seconds = ((self.total_items - total_progress) / items_per_second 
                          if items_per_second > 0 else 0)
            
            return {
                'global_progress': total_progress,
                'total_items': self.total_items,
                'progress_percent': (total_progress / self.total_items * 100) if self.total_items > 0 else 0,
                'items_per_second': items_per_second,
                'eta_seconds': eta_seconds,
                'elapsed_time': elapsed_time,
                'rank_progress': {p['rank']: p['local_progress'] for p in all_progress}
            }
        except Exception as e:
            logger.warning(f"Error syncing progress: {e}")
            return {
                'global_progress': self.local_progress,
                'total_items': self.total_items,
                'progress_percent': (self.local_progress / self.total_items * 100) if self.total_items > 0 else 0,
                'items_per_second': 0,
                'eta_seconds': 0,
                'elapsed_time': time.time() - self.start_time,
                'rank_progress': {self.rank: self.local_progress}
            }


class FileCoordinator:
    """Coordinates file operations across distributed processes"""
    
    def __init__(self, rank: int, world_size: int, output_dir: Path):
        self.rank = rank
        self.world_size = world_size
        self.output_dir = Path(output_dir)
        self.communicator = DistributedCommunicator(rank, world_size)
        
    def coordinate_file_creation(self, file_list: List[str]) -> List[str]:
        """Coordinate which files each rank should process"""
        # Distribute files round-robin
        assigned_files = []
        for i, file_path in enumerate(file_list):
            if i % self.world_size == self.rank:
                assigned_files.append(file_path)
        
        return assigned_files
    
    def collect_output_files(self) -> Dict[int, List[str]]:
        """Collect information about output files from all ranks"""
        # Get local output files
        local_files = []
        if self.output_dir.exists():
            local_files = [str(f) for f in self.output_dir.glob(f"*_gpu{self.rank}*")]
        
        file_info = {
            'rank': self.rank,
            'files': local_files,
            'count': len(local_files)
        }
        
        # Gather from all ranks
        all_file_info = self.communicator.all_gather_object(file_info)
        
        return {info['rank']: info['files'] for info in all_file_info}
    
    def ensure_output_directory(self):
        """Ensure output directory exists (coordinated across ranks)"""
        if self.rank == 0:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Wait for rank 0 to create directory
        self.communicator.barrier()


class MetricsAggregator:
    """Aggregates metrics across distributed processes"""
    
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.communicator = DistributedCommunicator(rank, world_size)
        
    def aggregate_metrics(self, local_metrics: Dict[str, Union[float, int]]) -> Optional[Dict[str, float]]:
        """Aggregate metrics across all ranks"""
        if self.rank == 0:
            # Collect metrics from all ranks
            all_metrics = self.communicator.all_gather_object(local_metrics)
            
            # Aggregate
            aggregated = {}
            for key in local_metrics.keys():
                values = [m[key] for m in all_metrics if key in m]
                if values:
                    if key.endswith('_count') or key.endswith('_total'):
                        aggregated[key] = sum(values)
                    elif key.endswith('_rate') or key.endswith('_avg'):
                        aggregated[key] = sum(values) / len(values)
                    else:
                        aggregated[key] = sum(values)  # Default: sum
            
            return aggregated
        else:
            # Non-rank-0 processes just send their metrics
            self.communicator.all_gather_object(local_metrics)
            return None


def log_distributed_info(rank: int, world_size: int, message: str):
    """Log information with rank prefix"""
    if rank == 0:
        logger.info(f"[DISTRIBUTED] {message}")
    else:
        logger.debug(f"[Rank {rank}] {message}")


def synchronize_random_seed(seed: int, rank: int, world_size: int) -> int:
    """Synchronize random seed across all ranks"""
    communicator = DistributedCommunicator(rank, world_size)
    
    if rank == 0:
        # Rank 0 decides the seed
        actual_seed = seed if seed is not None else int(time.time())
    else:
        actual_seed = None
    
    # Broadcast seed to all ranks
    actual_seed = communicator.broadcast_object(actual_seed, src_rank=0)
    
    return actual_seed


def wait_for_file_creation(file_path: Path, timeout_seconds: int = 300, rank: int = 0):
    """Wait for a file to be created (useful for coordination)"""
    start_time = time.time()
    
    while not file_path.exists():
        if time.time() - start_time > timeout_seconds:
            raise TimeoutError(f"Timeout waiting for file: {file_path}")
        time.sleep(1)
    
    if rank == 0:
        logger.info(f"File created: {file_path}")


def create_distributed_manifest(
    output_dir: Path,
    rank: int,
    world_size: int,
    local_results: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Create a distributed processing manifest"""
    
    communicator = DistributedCommunicator(rank, world_size)
    
    # Gather results from all ranks
    all_results = communicator.all_gather_object(local_results)
    
    if rank == 0:
        # Create consolidated manifest
        manifest = {
            'distributed_processing': {
                'world_size': world_size,
                'timestamp': time.time(),
                'total_files_processed': sum(r.get('files_processed', 0) for r in all_results),
                'total_samples': sum(r.get('total_samples', 0) for r in all_results),
                'processing_time': max(r.get('processing_time', 0) for r in all_results),
            },
            'rank_results': {i: result for i, result in enumerate(all_results)},
            'consolidation_status': 'completed'
        }
        
        # Save manifest
        manifest_path = output_dir / 'distributed_processing_manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"✅ Distributed manifest saved: {manifest_path}")
        return manifest
    
    return None