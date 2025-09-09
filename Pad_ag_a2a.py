import torch
import torch.nn as nn
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import os
import random

os.environ["ASCEND_LAUNCH_BLOCKING"] = "1"

def pad_tensor(self, input_tensor, target_len, dim=0, fill_value=0.0):
    if input_tensor.size(dim) >= target_len: return input_tensor
    pad_shape = list(input_tensor.shape)
    pad_shape[dim] = target_len - input_tensor.size(dim)
    return torch.cat([input_tensor, torch.full(pad_shape, fill_value, dtype=input_tensor.dtype, device=input_tensor.device)], dim=dim)

def crop_tensor(self, padded_tensor, original_len, dim=0):
    if original_len > padded_tensor.size(dim): 
        raise ValueError(f"原始长度{original_len}超过张量在维度{dim}上的长度{padded_tensor.size(dim)}")
    return padded_tensor.narrow(dim, 0, original_len)

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
        print(f"Rank {self.rank}: ✅ Accuracy verified within tolerance (atol={atol}, rtol={rtol})")
    
    def pad_tensor(self, input_tensor, target_len, dim=0, fill_value=0.0):
        if input_tensor.size(dim) >= target_len: return input_tensor
        pad_shape = list(input_tensor.shape)
        pad_shape[dim] = target_len - input_tensor.size(dim)
        return torch.cat([input_tensor, torch.full(pad_shape, fill_value, dtype=input_tensor.dtype, device=input_tensor.device)], dim=dim)

    def crop_tensor(self, padded_tensor, original_len, dim=0):
        if original_len > padded_tensor.size(dim): 
            raise ValueError(f"原始长度{original_len}超过张量在维度{dim}上的长度{padded_tensor.size(dim)}")
        return padded_tensor.narrow(dim, 0, original_len)
    def cleanup(self):
        """Clean up distributed resources"""
        dist.destroy_process_group()

"""
All-Gather + Column_Matmul + All-To-All
"""
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
        
        
        origin_size = hidden_states.size(0)
        pad_target = max(self.data_sizes)   
        hidden_states = self.pad_tensor(hidden_states, pad_target, dim=0)
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
        final_result = self.crop_tensor(final_result, origin_size, dim=0)
        
        torch.npu.synchronize()
        elapsed_ns = time.perf_counter_ns() - start_time
        
        return final_result, elapsed_ns



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
    for i in range(1):
        final_result, elapsed_ns = dist_linear.forward(hidden_states)
        print_time(i, elapsed_ns, rank)
    
    # Verify accuracy
    dist_linear.verify_accuracy(hidden_states, final_result)
    
    # Clean up
    dist_linear.cleanup()

def reduce_scatter(input_: torch.Tensor,
                    dim: int = -1, world_size: int = 1) -> torch.Tensor:
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input_
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}")

        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()

        # Note: This will produce an incorrect answer if we don't make
        # the input_tensor contiguous. Possible bug in reduce_scatter_tensor?
        input_tensor = input_.movedim(0, dim).contiguous()

        assert input_tensor.shape[0] % world_size == 0
        chunk_size = input_tensor.shape[0] // world_size
        output_shape = (chunk_size, ) + input_tensor.shape[1:]

        output_tensor = torch.empty(output_shape,
                                    dtype=input_tensor.dtype,
                                    device=input_tensor.device)

        # Perform reduce-scatter operation
        torch.distributed.reduce_scatter_tensor(output_tensor,
                                                input_tensor,
                                                )

        # Reshape before returning
        return output_tensor.movedim(0, dim).contiguous()

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
    world_size = 8

    data_sizes = [random.randint(10, 20) for _ in range(world_size)]
    # data_sizes = [24] * world_size
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
