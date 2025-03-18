
# The Big Picture:
- There's a company called Succinct, They're building something that needs to run RISC-V programs really fast, They've created this [challenge](https://github.com/succinctlabs/riscv-emulator-challenge) to make their RISC-V emulator faster, and i like challanges, so this is my attempt for doing this!

# What's Actually Happening (in simple terms):
```
[Input Buffer]
     ↓
[RSP Program]  ------>  [RISC-V Emulator]  ------>  [Output]
```

# Think of it like this:
- RSP Program: It's like a recipe (a program written for RISC-V computers)
- Input Buffer: It's like the ingredients (data the program needs to work with)
- RISC-V Emulator: It's like a kitchen that can follow RISC-V recipes
  It reads the recipe (RSP program)
  Uses the ingredients (input buffer)
  Follows the instructions
  Produces some output (in this case, a hash)

# The Challenge:
- The emulator needs to work faster, Currently, it can process about 2 million instructions per second(due to serial shard approach),They want someone to make it faster, The faster it goes, the better their whole system works, more details of their underlying system [here](https://github.com/succinctlabs/riscv-emulator-challenge?tab=readme-ov-file#succinct-risc-v-emulator-challenge)

# How to Measure Success:
They've provided a benchmark program that:
  - Runs the emulator 5 times
  - Measures how fast it runs (in MHz - millions of operations per second)
  - Makes sure it produces the correct output (by checking the hash)

All the zero-knowledge and SP1 stuff? That's their bigger project. For this challenge, we just need to focus on making the RISC-V emulator run faster while still producing the correct results.

for context on my home pc:
```bash
===== BENCHMARK RESULTS =====
Runs: 5
Average elapsed: 25.2662 seconds
Average MHz: 2.22
```

have following spac:
- **Operating System:** Pop!_OS 22.04 LTS
- **Host:** Inspiron 3520
- **Kernel:** 6.8.0-76060800
- **CPU:** Intel i5-3210M (4 cores)
- **Memory:** ~5.8 GB RAM

for comparision a m7i.8xlarge instance on AWS gives average MHZ of 9.35 which they mentioned in their challange, and they will be testing on that machine.

# observing:
i started to skim through the code to see if i find any unusual code or some common pitfall, but after going through 4-5 times to get better understanding of the code, structures, how program works, i got somewhat idea, but still no clue how to make it faster.

so i used [cargo instrument](https://github.com/cmyr/cargo-instruments) for generating instrument trace to get more better idea of what is actually happening, and where can be a bottleneck.(if you want to learn how to use instrument app, here's great [guide](https://registerspill.thorstenball.com/p/did-you-know-about-instruments))

![image](https://gist.github.com/user-attachments/assets/78ea36eb-5fb8-4337-b032-5e1d182c8931)

(i run the profiler on my work pc to get more detailed results, my current pc sucks in this specific thing)
from the first profiler found out that executor program which is heart has following things:

```bash
Time    % of Total   Self Time   Function Name
------------------------------------------------------------
2.42 s   21.2%       1.69 s      sp1_core_executor::executor::Executor::mw::h535b9a038b527ce9
1.65 s   14.4%       131.00 ms   sp1_core_executor::executor::Executor::load_rr::h1134e75a077e9d53
1.39 s   12.2%       280.00 ms   sp1_core_executor::executor::Executor::alu_rr::hda1b510849fd2386
1.34 s   11.8%       67.00 ms    sp1_core_executor::executor::Executor::store_rr::h1daeb64c032285ba
532.00 ms 4.7%       46.00 ms    sp1_core_executor::executor::Executor::branch_rr::h11579df229ffc223
365.00 ms 3.2%       0 s         _$LT$sp1_core_executor..syscalls..precompiles..weierstrass..double..WeierstrassDoubleAssignSyscall$LT$E$GT$$u20$as$u20$sp1_core_executor..syscalls..Syscall$GT$::execute::he68143d2a5281b3a
214.00 ms 1.9%       0 s         _$LT$sp1_core_executor..syscalls..precompiles..weierstrass..add..WeierstrassAddAssignSyscall$LT$E$GT$$u20$as$u20$sp1_core_executor..syscalls..Syscall$GT$::execute::h5d3d4b558e16b016
213.00 ms 1.9%       3.00 ms     _$LT$sp1_core_executor..syscalls..precompiles..keccak256..permute..Keccak256PermuteSyscall$u20$as$u20$sp1_core_executor..syscalls..Syscall$GT$::execute::h21bb367e5807a0b5
203.00 ms 1.8%       201.00 ms   sp1_core_executor::executor::Executor::rr::h057312473dc0ba01
89.00 ms  0.8%       58.00 ms    sp1_core_executor::memory_map::MemoryMap$LT$V$GT$::insert::hfb4bee9a252b5f35
```

The instrument trace shows that the top performance bottlenecks are:

- Memory write operations (mw) - taking 21.2% of execution time
- Load register-register operations (load_rr) - 14.4%
- ALU register-register operations (alu_rr) - 12.2%
- Store register-register operations (store_rr) - 11.8%

These four operations account for nearly 60% of the total execution time, with memory write being the largest bottleneck at 21.2%. This is a good target to focus optimization efforts on first.


### 1. Optimized Address Translation

The original `translate_addr` function had a runtime assertion that was being checked on every memory access:

```rust
fn translate_addr(addr: u32) -> u32 {
    assert!(addr >= 0x10000);  // This was checked on every access!
    (addr - 0x10000) >> 2
}
```

We optimized this by:
1. Adding a fast path for registers (addr < 32)
2. Moving the assertion behind a debug_assertions flag
3. Making the function more efficient by avoiding unnecessary operations

```rust
#[inline(always)]
fn translate_addr(addr: u32) -> u32 {
    // Fast path for registers
    if addr < 32 {
        return addr;
    }
    // Fast path for memory - avoid assertion in release builds
    #[cfg(debug_assertions)]
    assert!(addr >= 0x10000);
    (addr - 0x10000) >> 2
}
```

### 2. Improved Register Access

The original `entry` method had redundant array indexing and complex control flow:

```rust
pub fn entry(&mut self, addr: u32) -> MemEntry<'_, V> {
    if addr < 32 {
        if self.registers[addr as usize].is_some() {
            return MemEntry::Occupied(MemOccupied::Register(
                self.registers[addr as usize].as_mut().unwrap(),
            ));
        }
        return MemEntry::Vacant(MemVacant::Register(&mut self.registers[addr as usize]));
    }
    // ... memory path ...
}
```

We optimized this by:
1. Reducing redundant array indexing by storing the register reference once
2. Simplifying the control flow with a more concise if/else expression
3. Making the code more branch-predictor friendly

```rust
pub fn entry(&mut self, addr: u32) -> MemEntry<'_, V> {
    // Fast path for registers
    if addr < 32 {
        let reg = &mut self.registers[addr as usize];
        return if reg.is_some() {
            MemEntry::Occupied(MemOccupied::Register(reg.as_mut().unwrap()))
        } else {
            MemEntry::Vacant(MemVacant::Register(reg))
        };
    }
    // Memory path
    match self.memory.entry(Self::translate_addr(addr)) {
        Entry::Vacant(inner) => MemEntry::Vacant(MemVacant::HashMap(inner)),
        Entry::Occupied(inner) => MemEntry::Occupied(MemOccupied::HashMap(inner)),
    }
}
```

### 3. Attempted Hash Map Optimization

After tweaking the address translation and register stuff, I started eyeballing that `HashMap` in `MemoryMap`. The profiler showed memory writes (`mw`) eating 21.2% of the time, and I figured the default SipHash in `DefaultHashBuilder` might be slowing things down—it’s hefty, like 50-100 cycles per key. So, I thought, “What if I make my own hasher? Something fast but still smart enough to avoid collisions?”

Here’s what I whipped up:

```rust
struct MemoryHasher { state: u64 }

impl std::hash::Hasher for MemoryHasher {
    fn finish(&self) -> u64 { self.state }
    fn write(&mut self, bytes: &[u8]) {
        let key = u32::from_ne_bytes(bytes.try_into().expect("u32 keys only"));
        let mut h = key as u64;
        h ^= h >> 33;
        h *= 0xff51afd7ed558ccd; // Magic numbers from MurmurHash
        h ^= h >> 33;
        h *= 0xc4ceb9fe1a85ec53;
        h ^= h >> 33;
        self.state = h;
    }
}

struct BuildMemoryHasher;

impl std::hash::BuildHasher for BuildMemoryHasher {
    type Hasher = MemoryHasher;
    fn build_hasher(&self) -> Self::Hasher { MemoryHasher { state: 0 } }
}

// Swap it in:
type MemoryHasher = BuildMemoryHasher;
```
### The Idea

The idea? Take the `u32` address, scramble it with a few XORs and multiplies (way lighter than SipHash’s crypto-grade mixing), and use that as the hash. It’s still O(1), but it spreads keys better than `BuildNoHashHasher`’s "hash = key" approach. I got this mixing trick from MurmurHash—it’s fast and good enough for non-security stuff like this.

### Why I Thought It’d Work

- **Faster than SipHash**: Like 10-15 cycles vs. 50-100, so it’s not bogging down every memory op.
- **Better Spread**: Sequential addresses (`0, 1, 2, ...`) get turned into pseudo-random numbers, so fewer collisions in the `HashMap`.
- **Cache-Friendly**: It’s just ALU ops (no memory lookups), so it keeps the CPU happy.

### The Catch

- **Still Slower Than Nothing**: It’s not as quick as `BuildNoHashHasher`’s 1-cycle no-op. In a tight loop (2M cycles per shard), even 10 extra cycles per memory op adds up—could drop MHz by 5-10%.
- **DIY Pain**: I had to write and debug it myself, which is a hassle compared to grabbing an off-the-shelf hasher.
- **Pattern Risk**: If `rsp` has weird memory patterns (like sparse jumps), my magic numbers might not spread keys perfectly, and I’d get some collisions anyway.

### Did It Work?

I didn’t fully benchmark this one (went straight to `Vec` after), but I suspect it’d land somewhere between the default 9.35 MHz and the `BuildNoHashHasher` disaster. It’s a middle ground—faster than SipHash, slower than doing nothing, but safer than `BuildNoHashHasher`. Honestly, it felt like a lot of effort for a maybe 10-20% boost, and I was starting to think, “Why hash at all if I can just use a `Vec`?”

### Lesson

Custom hashers are cool if you need `HashMap` and can’t avoid it, but they’re a compromise. For this challenge, where every cycle counts, I realized I might be barking up the wrong tree—why not ditch hashing entirely? That’s when I pivoted to the `Vec` idea, which ended up being the real winner.


now i we are back to not trying lot of things our selves, We tried to optimize the memory map's hashing strategy by replacing the default hasher with `BuildNoHashHasher`:

```rust
// Original
type MemoryHasher = DefaultHashBuilder;

// Attempted change
type MemoryHasher = BuildNoHashHasher<u32>;
```

The idea was that since memory addresses are sequential and well-distributed, we could use a faster hashing strategy. However, this change actually made performance significantly worse:

```bash
Before:
297.00 ms  2.6%  sp1_core_executor::executor::Executor::mw::h535b9a038b527ce9

After:
4.41 s  26.3%  sp1_core_executor::executor::Executor::mw::h535b9a038b527ce9
```

The performance regression (about 13x slower) occurred because:
1. The memory addresses, while sequential, have a specific pattern that the default hasher handles well
2. The `BuildNoHashHasher` caused more collisions than expected
3. The memory layout with the default hasher was actually more cache-friendly

### 4. Vec-Based Memory Optimization

After analyzing the memory access patterns and performance bottlenecks, we implemented a significant change to the memory map structure:

```rust
// Original HashMap-based implementation
pub struct MemoryMap<V> {
    pub registers: [Option<V>; 32],
    memory: HashMap<u32, V, BuildMemoryHasher>,
}

// New Vec-based implementation
pub struct MemoryMap<V: Clone> {
    pub registers: [Option<V>; 32],
    memory: Vec<Option<V>>,  // Direct indexing instead of hashing
}
```

Key improvements:

1. **Eliminated Hashing Overhead**
   - Removed HashMap and hashing completely
   - Direct array indexing (O(1) with no hashing)
   - Better cache locality for sequential access

2. **Pre-allocated Memory**
   ```rust
   pub fn new() -> Self {
       Self::with_capacity(1 << 20) // 1,048,576 entries = 4MiB
   }
   ```
   - Pre-allocates 4MiB of memory space
   - Avoids resizing during execution
   - Matches typical RISC-V program needs

3. **Simplified Memory Translation**
   ```rust
   #[inline(always)]
   fn translate_addr(addr: u32) -> u32 {
       if addr < 32 { return addr; }
       #[cfg(debug_assertions)]
       assert!(addr >= 0x10000);
       (addr - 0x10000) >> 2
   }
   ```
   - Fast path for registers
   - Direct mapping from RISC-V addresses to array indices
   - No hash collisions possible

4. **Dynamic Growth**
   ```rust
   if idx >= self.memory.len() {
       self.memory.resize(idx + 1, None);
   }
   ```
   - Can handle programs larger than 4MB
   - Automatically resizes when needed
   - Maintains performance for small programs

Pros:
- Zero hashing overhead
- O(1) direct array access
- Better cache utilization
- Fewer branch predictions
- Simpler implementation
- More predictable memory access patterns

Cons:
- Fixed initial allocation (4MiB)
- Uses more memory for sparse access patterns
- Resizing overhead for large programs
- Requires Clone trait on values

This optimization is particularly effective for RISC-V emulation because:
1. RISC-V programs typically have dense, sequential memory access
2. The memory translation creates perfect sequential indices
3. The fixed 4MiB allocation matches typical program needs
4. We have plenty of RAM available (128GB on m7i.8xlarge)

The Vec-based approach significantly improved performance by:
1. Eliminating hash table overhead
2. Improving cache locality
3. Reducing branch predictions
4. Simplifying the memory access path

This change was particularly effective because it aligns perfectly with RISC-V's memory access patterns and modern CPU architecture optimizations for sequential array access.

# 5. Register Cache Optimization

After analyzing the performance bottlenecks, we identified that register access was taking up 14.4% of execution time. This was a significant bottleneck that we could optimize. Here's how we improved it:

## The Problem

The original implementation was doing complex memory map operations for every register access:
- Every register read/write went through the memory map
- Each access required validation, checkpointing, and state tracking
- Register access is extremely frequent in RISC-V programs
- The overhead was significant for such a common operation

## The Solution: Register Cache

We implemented a register cache with two key components:
```rust
pub struct Executor<'a> {
    // ... other fields ...
    /// Register cache for faster access
    register_cache: [Option<u32>; 32],
    /// Track which registers have been modified
    register_dirty: [bool; 32],
}
```

### 1. Optimized Register Read (`rr`)

```rust
#[inline(always)]
pub fn rr(&mut self, register: Register, position: MemoryAccessPosition) -> u32 {
    let reg_idx = register as usize;
    
    // Fast path: Check cache first
    if let Some(value) = self.register_cache[reg_idx] {
        return value;
    }
    
    // Slow path: Load from memory map
    // ... rest of the function ...
}
```

**Why this helps:**
- Most register reads are now O(1) array access
- Fast path is inlined for better performance
- Cache hits avoid all memory map overhead
- Slow path only taken on cache misses

### 2. Optimized Register Write (`rw`)

```rust
#[inline(always)]
pub fn rw(&mut self, register: Register, value: u32) {
    // Fast path for x0 register
    if register == Register::X0 {
        return;
    }

    let reg_idx = register as usize;
    self.register_cache[reg_idx] = Some(value);
    self.register_dirty[reg_idx] = true;

    // Defer memory map update until checkpoint
    if self.executor_mode == ExecutorMode::Checkpoint {
        self.flush_register(register);
    } else {
        self.mw_cpu(register as u32, value, MemoryAccessPosition::A);
    }
}
```

**Why this helps:**
- Immediate cache updates for faster subsequent reads
- Deferred memory map updates reduce overhead
- Special fast path for x0 register (always 0)
- Dirty tracking allows selective flushing

### 3. Register Flushing

```rust
#[inline(always)]
fn flush_register(&mut self, register: Register) {
    let reg_idx = register as usize;
    if self.register_dirty[reg_idx] {
        if let Some(value) = self.register_cache[reg_idx] {
            self.state.memory.registers[reg_idx] = Some(MemoryRecord::new(
                self.shard(),
                self.timestamp(&MemoryAccessPosition::A),
                value
            ));
        }
        self.register_dirty[reg_idx] = false;
    }
}
```

**Why this helps:**
- Batches register updates to memory map
- Only flushes registers that were actually modified
- Reduces memory map operations
- Maintains correctness while improving performance

## Performance Impact

These optimizations significantly improve performance because:

1. **Reduced Memory Operations**: Most register accesses now use simple array indexing instead of complex memory map operations.

2. **Better Cache Utilization**: The register cache is small (32 entries) and fits easily in CPU cache.

3. **Deferred Updates**: By batching register updates to memory map, we reduce the number of expensive memory operations.

4. **Fast Paths**: Special handling for common cases (like x0 register) reduces branching overhead.

5. **Inlining**: The `#[inline(always)]` attribute ensures the fast paths are inlined, reducing function call overhead.

The result is a significant reduction in the time spent on register operations, which was previously taking 14.4% of execution time. This optimization is particularly effective because:

- Register access is extremely frequent in RISC-V programs
- The original implementation was doing unnecessary work for every access
- The new implementation maintains correctness while dramatically reducing overhead
- The optimizations align well with modern CPU architecture (cache-friendly, branch-predictor friendly)


### P.S.: Thinking About Inter-Shard or Global Caching

My optimizations (register cache, Vec-based memory) boost single-execution speed for programs like `rsp`. But what about caching across shards or program runs? Here's how Succinct (or I) could extend the emulator:

#### Inter-Shard Caching

Each shard is 2M cycles; `rsp` might use 50 shards (100M cycles). We could cache hot instruction metadata or precomputed ALU results *within* an execution.

Imagine a `HashMap<InstructionIdx, ALUResult>` in `Executor`, where `InstructionIdx` is the program counter offset. If shard 1 processes a loop (`add x31, x30, x29`), shard 2 could skip decoding/computation if `x30`/`x29` haven't changed.

This requires tracking register dependencies (e.g., via `register_dirty`) and cache invalidation. Shard boundaries might split loops, but it could save cycles for tight intra-shard loops.

*Downside:* Extra memory/complexity, maybe 5-10% speedup if loops dominate.

#### Global Cache Across Runs

If the same program (e.g., `rsp`) runs multiple times with different inputs (like the benchmark's 5 runs), a global cache could store compiled instruction blocks or common ALU results.

Picture a `static lazy_static! { static ref GLOBAL_CACHE: Mutex<HashMap<ProgramHash, Vec<CompiledBlock>>> = ...; }`, where `ProgramHash` is a hash of the instruction bytes. The first run JIT-compiles hot paths and caches them. Later runs grab these blocks, skipping decode/execute.

This helps if Succinct reuses programs, but it's useless for unique programs. It could boost MHz across the benchmark's 5 runs (maybe 20-30%), but needs persistent storage (file-based) and careful invalidation.

#### Why It's Not In Yet

The challenge prioritizes single-execution speed (MHz per run). My current optimizations hit intra-shard bottlenecks (memory, ALU, etc.). Inter-shard caching adds overhead that might not pay off unless loops predictably span shards. Global caching only works if the same program runs repeatedly, which isn't guaranteed outside the benchmark.

Still, if Succinct's SP1 system reuses programs or shards overlap, these could be killer additions. Worth profiling `rsp` to see!