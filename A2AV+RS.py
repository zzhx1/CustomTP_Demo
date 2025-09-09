import torch
import torch.nn as nn
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
import time


class DistributedLinear:
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
        
        # Setup device and weight
        self.device = torch.device(f'npu:{rank}')
        self.weight = weight.to(self.device)

        # Ensure feature dimensions are divisible by world size
        assert inter_dim % world_size == 0, "inter_dim must be divisible by world size"
        self.chunk_size = inter_dim // world_size

        # Create local linear layer for this rank's chunk
        self.local_weight = self.weight[self.rank * self.chunk_size:(self.rank + 1) * self.chunk_size, :]  # [chunk_size, output_dim]
        self.local_inter_dim = self.local_weight.shape[0]  # chunk_size
        self.local_output_dim = self.local_weight.shape[1]  # output_dim

        self.local_oproj = nn.Linear(self.local_inter_dim, self.local_output_dim, bias=False)
        self.local_oproj.weight = nn.Parameter(self.local_weight.T)  # transpose for Linear layer
        self.local_oproj.to(self.device)

    def forward(self, hidden_states):
        """Perform the distributed forward pass with uneven all-to-all and reduce_scatter"""
        torch.npu.synchronize()
        start_time = time.perf_counter_ns()

        # Step 1: Uneven All-to-All on hidden_states (split along inter_dim)
        send_buf = torch.cat(torch.split(hidden_states, self.chunk_size, dim=1)).contiguous().view(-1)
        total_size = sum(self.data_sizes)
        recv_buf = torch.empty(total_size * self.chunk_size, dtype=hidden_states.dtype, device=self.device)

        send_counts = [hidden_states.size(0) * self.chunk_size] * self.world_size
        recv_counts = [size * self.chunk_size for size in self.data_sizes]

        dist.all_to_all_single(
            recv_buf,
            send_buf,
            output_split_sizes=recv_counts,
            input_split_sizes=send_counts
        )

        dist.barrier()

        # Reshape received data
        recv_chunks = []
        start_idx = 0
        for count in recv_counts:
            chunk = recv_buf[start_idx:start_idx + count]
            recv_chunks.append(chunk.view(-1, self.chunk_size))  # [recv_size, chunk_size]
            start_idx += count
        result = torch.cat(recv_chunks, dim=0)  # [total_size, chunk_size]
        assert result.shape == (total_size, self.chunk_size), f"Shape mismatch: {result.shape}"

        dist.barrier()

        # Step 2: Local Matmul
        local_o_result = self.local_oproj(result)  # [total_size, output_dim]
        assert local_o_result.shape[0] == total_size, "Shape mismatch after matmul"

        dist.barrier()

        # Step 3: ReduceScatter (uneven output split)
        recv_chunks = []
        start_idx = 0
        for size in self.data_sizes:
            chunk = local_o_result[start_idx:start_idx + size, :]  # [size, output_dim]
            recv_chunks.append(chunk.contiguous())
            start_idx += size

        final_result = torch.empty(self.data_sizes[self.rank], self.local_output_dim,
                                   dtype=local_o_result.dtype, device=self.device)

        dist.reduce_scatter(final_result, recv_chunks, op=dist.ReduceOp.SUM)

        torch.npu.synchronize()
        elapsed_ns = time.perf_counter_ns() - start_time

        return final_result, elapsed_ns

    def verify_accuracy(self, hidden_states, distributed_result):
        """Verify against full linear computation"""
        completed_o_proj = nn.Linear(
            self.inter_dim,
            self.output_dim,
            bias=False,
            dtype=torch.half,
            device=self.device
        )
        completed_o_proj.weight = nn.Parameter(self.weight.T)
        completed_o_proj_result = completed_o_proj(hidden_states)

        atol = 1e-3
        rtol = 1e-3
        torch.testing.assert_close(distributed_result, completed_o_proj_result, atol=atol, rtol=rtol)
        print(f"Rank {self.rank}: ✅ Accuracy verified within tolerance (atol={atol}, rtol={rtol})")
        
    def cleanup(self):
        """Clean up distributed resources"""
        dist.destroy_process_group()


def worker_process(rank, world_size, data_sizes, weight, inter_dim, output_dim):
    """Worker process function for each rank"""
    print(f"Rank {rank} starting...")

    # Create distributed linear layer
    dist_linear = DistributedLinear(rank, world_size, inter_dim, output_dim, weight, data_sizes)

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
    # Configuration parameters —— 注意：这里还原原始参数！
    world_size = 4
    data_sizes = [8, 9, 10, 11]  # 不均匀输入
    inter_dim = 16384            # 原始输入维度
    output_dim = 7168            # 原始输出维度
    weight = torch.randn((inter_dim, output_dim), dtype=torch.half)

    print(f"#### START TEST COMMUNICATION ####")

    mp.spawn(
        worker_process,
        args=(world_size, data_sizes, weight, inter_dim, output_dim),
        nprocs=world_size,
        join=True
    )