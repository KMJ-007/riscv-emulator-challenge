use serde::{Deserialize, Serialize};

/// Memory map optimized for RISC-V emulation.
/// 
/// Uses a Vec for direct indexing of memory locations, with a separate array for registers.
/// Pre-allocates 4MiB of memory space (1,048,576 entries) to avoid resizing.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct MemoryMap<V: Clone> {
    /// Register values (0-31)
    pub registers: [Option<V>; 32],
    /// Memory values, indexed by (addr - 0x10000) >> 2
    memory: Vec<Option<V>>,
}

impl<V: Clone> MemoryMap<V> {
    /// Creates a new memory map with pre-allocated space for 4MiB of memory
    pub fn new() -> Self {
        Self::with_capacity(1 << 20) // 1,048,576 entries = 4MiB
    }

    /// Creates a new memory map with specified capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            memory: vec![None; capacity],
            registers: [const { None }; 32],
        }
    }

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

    /// Gets a reference to a value at the given address
    pub fn get(&self, addr: &u32) -> Option<&V> {
        if *addr < 32 {
            return self.registers[*addr as usize].as_ref();
        }
        let idx = Self::translate_addr(*addr) as usize;
        self.memory.get(idx).and_then(|v| v.as_ref())
    }

    /// Gets a mutable reference to a value at the given address
    pub fn get_mut(&mut self, addr: &u32) -> Option<&mut V> {
        if *addr < 32 {
            return self.registers[*addr as usize].as_mut();
        }
        let idx = Self::translate_addr(*addr) as usize;
        self.memory.get_mut(idx).and_then(|v| v.as_mut())
    }

    /// Returns an entry for the given address
    pub fn entry(&mut self, addr: u32) -> MemEntry<'_, V> {
        if addr < 32 {
            let reg = &mut self.registers[addr as usize];
            return if reg.is_some() {
                MemEntry::Occupied(MemOccupied::Register(reg.as_mut().unwrap()))
            } else {
                MemEntry::Vacant(MemVacant::Register(reg))
            };
        }

        let idx = Self::translate_addr(addr) as usize;
        // Ensure capacity
        if idx >= self.memory.len() {
            self.memory.resize(idx + 1, None);
        }

        let value = &mut self.memory[idx];
        if value.is_some() {
            MemEntry::Occupied(MemOccupied::Vec(value.as_mut().unwrap()))
        } else {
            MemEntry::Vacant(MemVacant::Vec(value))
        }
    }

    /// Inserts a value at the given address
    pub fn insert(&mut self, addr: u32, value: V) -> Option<V> {
        if addr < 32 {
            return std::mem::replace(&mut self.registers[addr as usize], Some(value));
        }
        let idx = Self::translate_addr(addr) as usize;
        // Ensure capacity
        if idx >= self.memory.len() {
            self.memory.resize(idx + 1, None);
        }
        std::mem::replace(&mut self.memory[idx], Some(value))
    }

    /// Removes a value at the given address
    pub fn remove(&mut self, addr: &u32) -> Option<V> {
        if *addr < 32 {
            return std::mem::replace(&mut self.registers[*addr as usize], None);
        }
        let idx = Self::translate_addr(*addr) as usize;
        if idx < self.memory.len() {
            std::mem::replace(&mut self.memory[idx], None)
        } else {
            None
        }
    }

    /// Returns the inner memory vector
    pub fn into_inner(self) -> Vec<Option<V>> {
        self.memory
    }
}

/// Entry for a memory location
pub enum MemEntry<'a, V> {
    /// Occupied entry
    Occupied(MemOccupied<'a, V>),
    /// Vacant entry
    Vacant(MemVacant<'a, V>),
}

/// Occupied memory entry
pub enum MemOccupied<'a, V> {
    /// Register entry
    Register(&'a mut V),
    /// Vec entry
    Vec(&'a mut V),
}

/// Vacant memory entry
pub enum MemVacant<'a, V> {
    /// Register entry
    Register(&'a mut Option<V>),
    /// Vec entry
    Vec(&'a mut Option<V>),
}

impl<'a, V> MemEntry<'a, V> {
    pub fn or_insert_with<F>(self, default: F) -> &'a mut V
    where
        F: FnOnce() -> V,
    {
        match self {
            Self::Vacant(vacant) => vacant.insert(default()),
            Self::Occupied(occupied) => occupied.into_mut(),
        }
    }

    pub fn or_insert(self, default: V) -> &'a mut V {
        self.or_insert_with(|| default)
    }

    pub fn and_modify<F>(self, f: F) -> MemEntry<'a, V>
    where
        F: FnOnce(&mut V),
    {
        match self {
            Self::Vacant(vacant) => Self::Vacant(vacant),
            Self::Occupied(mut occupied) => {
                occupied.modify(f);
                Self::Occupied(occupied)
            }
        }
    }
}

impl<'a, V> MemOccupied<'a, V> {
    pub fn get(&self) -> &V {
        match self {
            Self::Register(v) => v,
            Self::Vec(v) => v,
        }
    }

    pub fn into_mut(self) -> &'a mut V {
        match self {
            Self::Register(v) => v,
            Self::Vec(v) => v,
        }
    }

    pub fn modify<F>(&mut self, f: F)
    where
        F: FnOnce(&mut V),
    {
        match self {
            Self::Register(v) => f(v),
            Self::Vec(v) => f(v),
        }
    }
}

impl<'a, V> MemVacant<'a, V> {
    pub fn insert(self, value: V) -> &'a mut V {
        match self {
            Self::Register(opt) => opt.insert(value),
            Self::Vec(opt) => opt.insert(value),
        }
    }
}