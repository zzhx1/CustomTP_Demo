import torch
import torch.nn as nn
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
import time


class DistributedLinearBase:
    def __init__(self, rank, world_size, inter_dim, output_dim, weight, data_sizes):
        self.rank = rank
        self.world_size = world_size
        self.inter_dim = inter_dim
        self.output_dim = output_dim
        self.data_sizes = data_sizes
        
        # Initialize process group
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
        """Verify the distributed computation matches the full computation"""
        # Create reference full linear layer
        completed_o_proj = nn.Linear(
            self.inter_dim, 
            self.output_dim, 
            bias=False, 
            dtype=torch.half, 
            device=self.device
        )
        completed_o_proj.weight = nn.Parameter(self.weight.T)
        reference_result = completed_o_proj(hidden_states)
        
        # Compare results
        atol = 1e-3
        rtol = 1e-3
        torch.testing.assert_close(distributed_result, reference_result, atol=atol, rtol=rtol)
    
    def cleanup(self):
        """Clean up distributed resources"""
        dist.destroy_process_group()


class Distributed_Column_Linear(DistributedLinearBase):
    def __init__(self, rank, world_size, inter_dim, output_dim, weight, data_sizes):
        super().__init__(rank, world_size, inter_dim, output_dim, weight, data_sizes)
        self.local_output_dim = output_dim // world_size
        self.local_o_proj = self._create_local_linear()
        
    def _create_local_linear(self):
        """Create the local linear projection for this rank"""
        local_o_proj = nn.Linear(
            in_features=self.inter_dim, 
            out_features=self.local_output_dim, 
            bias=False, 
            dtype=torch.half, 
            device=self.device
        )
        local_weight = self.weight[:, self.rank*self.local_output_dim : (self.rank+1)*self.local_output_dim].npu()
        local_o_proj.weight = nn.Parameter(local_weight.T)
        return local_o_proj
    
    def forward(self, hidden_states):
        """Perform the distributed forward pass"""
        torch.npu.synchronize()
        start_time = time.perf_counter_ns()
        local_batch_size = hidden_states.size(0)
        
        # All gather input from all ranks
        gathered_input = torch.empty(
            self.world_size * hidden_states.size(0),  # total batch size
            hidden_states.size(1),
            dtype=hidden_states.dtype,
            device='npu'
        )
        dist.all_gather_into_tensor(gathered_input, hidden_states)

        # Local matrix multiplication
        local_o_result = self.local_o_proj(gathered_input)
        
        ######## All-to-all communication
        output_dim_per_rank = self.local_output_dim
        all_to_all_result = torch.empty(
            local_batch_size * self.output_dim,
            dtype=local_o_result.dtype,
            device='npu'
        )
        input_split_sizes = [local_batch_size] * self.world_size
        output_split_sizes = [local_batch_size * output_dim_per_rank] * self.world_size

        dist.all_to_all_single(
            all_to_all_result,
            local_o_result.contiguous(),
            output_split_sizes,
            input_split_sizes
        )
        reshaped = all_to_all_result.view(self.world_size, local_batch_size, output_dim_per_rank)
        final_result = reshaped.permute(1, 0, 2).reshape(local_batch_size, -1)        
        
        torch.npu.synchronize()
        elapsed_ns = time.perf_counter_ns() - start_time
        
        return final_result, elapsed_ns


class Distributed_Row_Linear(DistributedLinearBase):
    def __init__(self, rank, world_size, inter_dim, output_dim, weight, data_sizes):
        super().__init__(rank, world_size, inter_dim, output_dim, weight, data_sizes)
        self.local_inter_dim = inter_dim // world_size
        self.local_o_proj = self._create_row_linear()
        
    def _create_row_linear(self):
        """Create the local linear projection for this rank"""
        local_o_proj = nn.Linear(
            in_features=self.local_inter_dim, 
            out_features=self.output_dim, 
            bias=False, 
            dtype=torch.half, 
            device=self.device
        )
        local_weight = self.weight[self.rank*self.local_inter_dim : (self.rank+1)*self.local_inter_dim, :].npu()
        local_o_proj.weight = nn.Parameter(local_weight)
        return local_o_proj
    
    def forward(self, hidden_states):
        """Perform the distributed forward pass for row parallelism"""
        torch.npu.synchronize()
        start_time = time.perf_counter_ns()
        
        # Split input along the feature dimension
        input_splits = torch.split(hidden_states, self.local_inter_dim, dim=1)
        local_input = input_splits[self.rank]
        
        # Local matrix multiplication
        local_output = self.local_o_proj(local_input)
        
        # All-reduce across all ranks
        output = torch.zeros_like(local_output)
        dist.all_reduce(local_output, output=output, op=dist.ReduceOp.SUM)
        
        torch.npu.synchronize()
        elapsed_ns = time.perf_counter_ns() - start_time
        
        return output, elapsed_ns


def worker_process(rank, world_size, data_sizes, weight, inter_dim, output_dim):
    """Worker process function for each rank"""
    print(f"Rank {rank} starting...")
    
    # Create distributed linear layer
    dist_linear = Distributed_Column_Linear(rank, world_size, inter_dim, output_dim, weight, data_sizes)
    # Or use row linear:
    # dist_linear = Distributed_Row_Linear(rank, world_size, inter_dim, output_dim, weight, data_sizes)
    
    # Generate input data
    hidden_states = torch.randn(data_sizes[rank], inter_dim, device=dist_linear.device)
    
    # Perform distributed forward pass
    for i in range(10):
        final_result, elapsed_ns = dist_linear.forward(hidden_states)
        print_time(i, elapsed_ns, rank)
    
    # Verify accuracy
    dist_linear.verify_accuracy(hidden_states, final_result)
    
    # Clean up
    dist_linear.cleanup()

def print_time(iter, elapsed_ns, rank):
    # Print timing information
    if rank == 0:
        print(f"iteration {iter}")
    torch.npu.synchronize()
    if elapsed_ns >= 1_000_000:  # >=1ms
        elapsed_str = f"{elapsed_ns/1_000_000:.3f} ms"
    else:
        elapsed_str = f"{elapsed_ns/1000:.3f} us"
    print(f"    Rank {rank}: Communication time is {elapsed_str}")

if __name__ == "__main__":
    # Configuration parameters
    world_size = 4
    data_sizes = [24, 24, 24, 24]
    inter_dim = 16384
    output_dim = 7168
    weight = torch.randn((inter_dim, output_dim), dtype=torch.half)

    print(f"#### START TEST COMMUNICATION ####")

    mp.spawn(
        worker_process,
        args=(world_size, data_sizes, weight, inter_dim, output_dim),
        nprocs=world_size,
        join=True
    )