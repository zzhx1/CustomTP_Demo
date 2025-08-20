import torch
import torch.nn as nn
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
import time


class DistributedLinearBase:
    """
    Base class for distributed linear layers.
    Handles common initialization and utility methods.
    """
    def __init__(self, rank, world_size, inter_dim, output_dim, weight, data_sizes):
        self.rank = rank
        self.world_size = world_size
        self.inter_dim = inter_dim
        self.output_dim = output_dim
        self.data_sizes = data_sizes
        
        # Initialize HCCL process group for NPU communication
        dist.init_process_group(
            backend='hccl',
            init_method='tcp://127.0.0.1:55223',
            world_size=world_size,
            rank=rank
        )
        torch.npu.set_device(rank)
        self.device = torch.device(f'npu:{rank}')
        self.weight = weight.to(self.device)
        
    def verify_accuracy(self, hidden_states, distributed_result):
        """
        Verify that distributed computation matches single-device computation.
        
        Args:
            hidden_states: Input tensor
            distributed_result: Result from distributed computation
        """
        # Create reference full linear layer on single device
        full_linear = nn.Linear(
            self.inter_dim, 
            self.output_dim, 
            bias=False, 
            dtype=torch.half, 
            device=self.device
        )
        full_linear.weight = nn.Parameter(self.weight.T)
        reference_result = full_linear(hidden_states)
        
        # Compare results with tolerance
        atol = 1e-3
        rtol = 1e-3
        torch.testing.assert_close(distributed_result, reference_result, atol=atol, rtol=rtol)
        print(f"Rank {self.rank}: Accuracy verification passed!")
    
    def cleanup(self):
        """Clean up distributed process group resources."""
        dist.destroy_process_group()





class Distributed_Row_Linear(DistributedLinearBase):
    """
    Distributed linear layer with All-to-All + Reduce-Scatter communication pattern.
    Uses column parallelism with optimized A2A and RS operations.
    """
    def __init__(self, rank, world_size, inter_dim, output_dim, weight, data_sizes):
        super().__init__(rank, world_size, inter_dim, output_dim, weight, data_sizes)
        self.local_input_dim = inter_dim // world_size
        self.local_o_proj = self._create_local_linear()
        
    def _create_local_linear(self):
        """Create local linear projection with sliced weights for column parallelism."""
        local_linear = nn.Linear(
            in_features=self.local_input_dim, 
            out_features=self.output_dim, 
            bias=False, 
            dtype=torch.half, 
            device=self.device
        )
        # Slice weights for this rank's portion of input dimension
        weight_slice = self.weight[self.rank*self.local_input_dim : (self.rank+1)*self.local_input_dim, :]
        local_linear.weight = nn.Parameter(weight_slice.T)
        return local_linear
    
    def forward(self, hidden_states):
        """
        Perform distributed forward pass with A2A + Reduce-Scatter pattern.
        
        Steps:
        1. Reshape and prepare input for all-to-all communication
        2. Perform all-to-all to distribute input chunks
        3. Local matrix multiplication on each rank
        4. Reduce-scatter to aggregate and distribute results
        
        Returns:
            final_result: Output tensor
            elapsed_ns: Execution time in nanoseconds
        """
        torch.npu.synchronize()
        start_time = time.perf_counter_ns()
        local_batch_size = hidden_states.size(0)
        
        # Step 1: Prepare input for all-to-all communication
        # Reshape input to (local_batch_size * world_size, local_input_dim)
        total_batch_size = local_batch_size * self.world_size
        split_input = hidden_states.reshape(total_batch_size, -1)
        
        # Split input into chunks for each rank
        split_input_chunks = split_input.chunk(self.world_size, dim=0)
        send_tensor = torch.stack(split_input_chunks)  # Shape: (world_size, local_batch_size, local_input_dim)
        
        # Step 2: Perform all-to-all communication
        recv_tensor = torch.empty_like(send_tensor)
        dist.all_to_all(recv_tensor, send_tensor)
        
        # Reshape received tensor for local computation
        # Transpose and reshape to (total_batch_size, local_input_dim)
        recv_reshaped = recv_tensor.transpose(0, 1).reshape(-1, self.local_input_dim)
        
        # Step 3: Local matrix multiplication
        local_o_result = self.local_o_proj(recv_reshaped)  # Shape: (total_batch_size, output_dim)
        
        # Step 4: Reduce-scatter to aggregate results
        final_result = torch.empty(
            local_batch_size, 
            self.output_dim,
            dtype=local_o_result.dtype,
            device=self.device
        )
        
        # Split local result into chunks for reduce-scatter
        input_chunks = list(local_o_result.chunk(self.world_size, dim=0))
        dist.reduce_scatter(output=final_result, input_list=input_chunks, op=dist.ReduceOp.SUM)
        
        torch.npu.synchronize()
        elapsed_ns = time.perf_counter_ns() - start_time
        
        return final_result, elapsed_ns


def worker_process(rank, world_size, data_sizes, weight, inter_dim, output_dim):
    """
    Worker process function for each rank.
    
    Args:
        rank: Process rank (0 to world_size-1)
        world_size: Total number of processes
        data_sizes: Batch sizes for each rank
        weight: Weight matrix for linear layer
        inter_dim: Input dimension
        output_dim: Output dimension
    """
    print(f"Rank {rank} starting with batch size {data_sizes[rank]}...")
    
    
    dist_linear = Distributed_Row_Linear(rank, world_size, inter_dim, output_dim, weight, data_sizes)
    
    # Generate input data
    hidden_states = torch.randn(data_sizes[rank], inter_dim, device=dist_linear.device, dtype=torch.half)
    
    # Perform multiple distributed forward passes for benchmarking
    for i in range(10):
        final_result, elapsed_ns = dist_linear.forward(hidden_states)
        print_time(i, elapsed_ns, rank)
    
    # Verify computation accuracy
    dist_linear.verify_accuracy(hidden_states, final_result)
    
    # Clean up resources
    dist_linear.cleanup()
    print(f"Rank {rank} completed successfully.")


def print_time(iteration, elapsed_ns, rank):
    """
    Print timing information in appropriate units.
    
    Args:
        iteration: Current iteration number
        elapsed_ns: Elapsed time in nanoseconds
        rank: Process rank
    """
    torch.npu.synchronize()
    
    # Only rank 0 prints iteration info
    if rank == 0:
        print(f"Iteration {iteration}:")
    
    # Convert to appropriate time unit
    if elapsed_ns >= 1_000_000:  # >=1ms
        elapsed_str = f"{elapsed_ns/1_000_000:.3f} ms"
    else:
        elapsed_str = f"{elapsed_ns/1000:.3f} us"
    
    print(f"    Rank {rank}: Execution time: {elapsed_str}")


if __name__ == "__main__":
    # Configuration parameters for distributed training
    world_size = 4
    data_sizes = [24, 24, 24, 24]  # Batch sizes for each rank
    inter_dim = 16384               # Input dimension
    output_dim = 7168               # Output dimension
    weight = torch.randn((inter_dim, output_dim), dtype=torch.half)

    print(f"#### START DISTRIBUTED LINEAR LAYER TEST (A2A+RS) ####")
    print(f"World size: {world_size}, Input dim: {inter_dim}, Output dim: {output_dim}")

    # Launch distributed processes
    mp.spawn(
        worker_process,
        args=(world_size, data_sizes, weight, inter_dim, output_dim),
        nprocs=world_size,
        join=True
    )