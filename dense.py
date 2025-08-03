import torch
import torch.nn as nn
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import os 

os.environ['ASCEND_LAUNCH_BLOCKING'] = '1'

class MLPLinear:
    def __init__(self, rank, world_size, data_sizes, weight_up, weight_down, hidden_dim, inter_dim):
        self.rank = rank
        self.world_size = world_size
        self.hidden_dim = hidden_dim
        self.inter_dim = inter_dim
        self.data_sizes = data_sizes
        self.local_hidden_dim = hidden_dim // world_size
        self.local_inter_dim = inter_dim // world_size

        # Initialize process group
        dist.init_process_group(
            backend='hccl',
            init_method='tcp://127.0.0.1:55223',
            world_size=world_size,
            rank=rank
        )
        torch.npu.set_device(rank)
        
        # Setup local linear layer
        self.device = torch.device(f'npu:{rank}')
        self.weight_up = weight_up.to(self.device)
        self.weight_down = weight_down.to(self.device)

        self.gate_up_proj = self._create_local_linear(self.weight_up, "up")
        self.down_proj = self._create_local_linear(self.weight_down, "down")
        
    def _create_local_linear(self, weight, direction):
        """Create the local linear projection for this rank"""

        if direction == "up":
            self.local_output_dim = self.local_inter_dim  
            in_features = self.hidden_dim
            out_features = self.local_output_dim
        elif direction == "down":
            self.local_output_dim = self.local_hidden_dim
            in_features = self.inter_dim
            out_features = self.local_output_dim

        local_linear = nn.Linear(
            in_features=in_features, 
            out_features=out_features, 
            bias=False, 
            dtype=torch.half, 
            device=self.device
        )
        # Extract the local weight slice for this rank
        if direction == "up":
            local_weight = weight[:, self.rank*self.local_output_dim : (self.rank+1)*self.local_output_dim].npu()
            print(f"rank:{self.rank}: up_weight: {local_weight.shape} ")
        elif direction == "down":
            local_weight = weight[self.rank*self.local_inter_dim : (self.rank+1)*self.local_inter_dim, :].npu()
            print(f"rank:{self.rank}: down_weight: {local_weight.shape} ")
        local_linear.weight = nn.Parameter(local_weight.T)
        return local_linear
    
    
    def forward(self, hidden_states):
        """Perform the distributed forward pass"""
        torch.npu.synchronize()
        start_time = time.perf_counter_ns()
        local_batch_size = hidden_states.size(0)
        # All gather input from all ranks
        gathered_input = [torch.empty(batch_size, hidden_states.size(1), 
                                    dtype=hidden_states.dtype, device='npu') 
                         for batch_size in self.data_sizes]
    
        dist.all_gather(gathered_input, hidden_states)
        complete_input = torch.cat(gathered_input, dim=0)

        # Local matrix multiplication for the up projection
        gate_up_result = self.gate_up_proj(complete_input)

        print(f"rank:{dist.get_rank()}: gate_up_result: {gate_up_result.shape} ")
        # Perform local matrix multiplication for the down projection
        down_result = self.down_proj(gate_up_result)
        print(f"rank:{dist.get_rank()}: down_result: {down_result.shape} ")

        final_result = torch.empty(
            down_result.size(0),
            self.hidden_dim,
            dtype=down_result.dtype, device='npu'
        )
        # 将down_result转换为列表
        output_list = [torch.empty_like(final_result) for _ in range(self.world_size)]
        output_list[self.rank] = down_result

        dist.reduce_scatter(final_result, output_list, op=dist.ReduceOp.SUM)
        final_result = final_result[self.rank*local_batch_size : (self.rank+1)*local_batch_size, :]
        
        torch.npu.synchronize()
        elapsed_ns = time.perf_counter_ns() - start_time
        
        return final_result, elapsed_ns
    
    def verify_accuracy(self, hidden_states, distributed_result):
        """Verify the distributed computation matches the full computation"""

        completed_up_proj = nn.Linear(
            self.hidden_dim, 
            self.inter_dim, 
            bias=False, 
            dtype=torch.half, 
            device=self.device
        )
        completed_up_proj.weight = nn.Parameter(self.weight_up.T)
        complete_down_proj = nn.Linear(
            self.inter_dim, 
            self.hidden_dim, 
            bias=False, 
            dtype=torch.half, 
            device=self.device
        )
        complete_down_proj.weight = nn.Parameter(self.weight_down.T)

        Intermediate_results = completed_up_proj(hidden_states)
        reference_result = complete_down_proj(Intermediate_results)
        
        # Compare results
        atol = 1e-3
        rtol = 1e-3
        torch.testing.assert_close(distributed_result, reference_result, atol=atol, rtol=rtol)
        print(f"Rank {self.rank} results verified successfully!")

    def cleanup(self):
        """Clean up distributed resources"""
        dist.destroy_process_group()


def worker_process(rank, world_size, data_sizes, weight_up, weight_down, hidden_dim, inter_dim):
    """Worker process function for each rank"""
    print(f"Rank {rank} starting...")
    
    # Create distributed linear layer
    dist_linear = MLPLinear(rank,  world_size, data_sizes, weight_up, weight_down, hidden_dim, inter_dim)
    
    # Generate input data
    hidden_states = torch.randn(data_sizes[rank], hidden_dim, device=dist_linear.device)
    
    # Perform distributed forward pass
    for i in range(1):

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
    hidden_dim = 7168
    inter_dim = 18432 
    weight_up = torch.randn((hidden_dim, inter_dim), dtype=torch.half)
    weight_down = torch.randn((inter_dim, hidden_dim), dtype=torch.half)

    print(f"#### START TEST COMMUNICATION ####")

    mp.spawn(
        worker_process,
        args=(world_size, data_sizes, weight_up, weight_down, hidden_dim, inter_dim),
        nprocs=world_size,
        join=True
    )
