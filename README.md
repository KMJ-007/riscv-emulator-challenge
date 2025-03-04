# Succinct RISC-V Emulator Challenge

Succinct is building a SP1, a zero-knowledge virtual machine that can prove the execution of RISC-V bytecode.

**RISC-V emulator performance is critical for proving latency.** Since SP1 distributes proving workloads across a GPU cluster, the primary bottleneck is how quickly we can generate work for these GPUs. This process begins with executing RISC-V bytecode, which is inherently serial and limits overall throughput. Each unit of work, called a "shard," represents 2 million RISC-V cycles of execution. For example, a 100 million cycle execution would be split into 50 shards. Our goal is to optimize the RISC-V emulator’s performance to efficiently feed the GPUs and maximize parallelism.

```
Time ─────────────────────────────────────────────────▶

   Execution (Serial, Emulator)
   +---------+---------+---------+---------+---------+
   | Shard 1 | Shard 2 | Shard 3 | Shard 4 | Shard 5 |  
   +---------+---------+---------+---------+---------+
       │         │         │         │         │
       ▼         ▼         ▼         ▼         ▼
   Proving (Parallel, GPUs)
   ───────▶[GPU 1: Shard 1 Proof]────────▶
             ───────▶[GPU 2: Shard 2 Proof]────────▶
                       ───────▶[GPU 3: Shard 3 Proof]────────▶
                                 ───────▶[GPU 4: Shard 4 Proof]────────▶
                                           ───────▶[GPU 5: Shard 5 Proof]────────▶
```

Succinct is seeking for new approaches to this problem and outstanding engineers to work on it. If you have a solution, please email riscv-emulator-challenge@succinct.xyz with a link to your GitHub repository or zip file.

## Task

We’ve created a basic starter RISC-V emulator in Rust [here](https://github.com/succinctlabs/riscv-emulator-challenge) alongside a basic benchmarking script. Your task is to optimize the performance of this implementation on the `rsp` program and maximize the `Average MHz` statistic.

To benchmark the performance, run the following command:

```
cd benchmark
cargo run —-release 
```

Start by exploring the `crates/executor` crate to understand its current implementation and identify performance bottlenecks. Focus on improving the existing implementation while ensuring that any modifications are benchmarked for performance improvements and correctness.

Note that performance varies based on the hardware being utilized. Submissions will be judged using a m7i.8xlarge instance on AWS. With the existing implementation, the average MHz is around 9.35.

## Leaderboard

Submissions will be continuously evaluated and a leaderboard will be maintained.