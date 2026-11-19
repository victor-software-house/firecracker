// Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Portions Copyright 2017 The Chromium OS Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the THIRD-PARTY file.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Barrier, Mutex, MutexGuard};

#[cfg(target_arch = "x86_64")]
use kvm_bindings::KVM_IRQCHIP_IOAPIC;
use kvm_bindings::{
    KVM_IRQ_ROUTING_IRQCHIP, KVM_IRQ_ROUTING_MSI, KVM_MSI_VALID_DEVID, KvmIrqRouting,
    kvm_irq_routing_entry, kvm_userspace_memory_region,
};
use kvm_ioctls::VmFd;
use serde::{Deserialize, Serialize};
use userfaultfd::Uffd;
use vm_memory::GuestMemoryBackend;
use vmm_sys_util::errno;
use vmm_sys_util::eventfd::EventFd;
use vmm_sys_util::terminal::Terminal;

use crate::arch::{GSI_MSI_END, host_page_size};
pub use crate::arch::{KvmVm, KvmVmError, VmState};
use crate::logger::{debug, info, warn};
use crate::nitro_enclave::enclave_vm::EnclaveVm;
use crate::pci::PciSBDF;
use crate::pci::msix::MsixTableEntry;
use crate::persist::CreateSnapshotError;
use crate::vmm_config::snapshot::SnapshotType;
use crate::vstate::bus::Bus;
use crate::vstate::interrupts::{InterruptError, MsixVector, MsixVectorGroup};
use crate::vstate::kvm::Kvm;
use crate::vstate::memory::{
    EvictWrite, GuestMemoryExtension, GuestMemoryMmap, GuestMemoryRegion, GuestMemoryState,
    GuestRegionMmap, GuestRegionMmapExt, MemoryError,
};
use crate::vstate::resources::ResourceAllocator;
use crate::vstate::vcpu::{StartThreadedError, VcpuError, VcpuHandle};
use crate::{DirtyBitmap, Vcpu, mem_size_mib};

/// Maximum number of vCPUs supported by a KVM microVM.
///
/// The value is arbitrary, and far below what KVM allows (1024 on x86_64, 512 on aarch64).
/// Raising it hits, on x86_64: Intel's CPUID leaf 0x4 EAX[31:26] encodes the core ID in 6 bits, so
/// 64 cores; CPUID leaf 0x1 EBX[23:16] rejects counts above 128; and the MP table and ACPI MADT
/// both identify a CPU by an 8-bit APIC ID. On aarch64 none of these exist, and the bound is the
/// kernel VGIC, at 512 vCPUs.
pub const MAX_SUPPORTED_KVM_VCPUS: u16 = 32;

/// The KVM vCPU plumbing narrows the count to a `u8`, so the limit must fit in
/// one for that narrowing to be infallible.
const _: () = assert!(MAX_SUPPORTED_KVM_VCPUS <= u8::MAX as u16);

/// Error type for [`KvmVm::start_vcpus`].
#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum StartVcpusError {
    /// Failed to set terminal mode: {0}
    SetTerminalMode(#[from] vmm_sys_util::errno::Error),
    /// Vcpu handle error: {0}
    VcpuHandle(#[from] StartThreadedError),
}

#[derive(Debug, Serialize, Deserialize)]
/// A struct representing an interrupt line used by some device of the microVM
pub struct RoutingEntry {
    entry: kvm_irq_routing_entry,
    masked: bool,
}

/// Architecture independent parts of a VM.
#[derive(Debug)]
pub struct VmCommon {
    /// The KVM file descriptor used to access this KvmVm.
    pub fd: VmFd,
    max_memslots: u32,
    /// The guest memory of this KvmVm.
    pub guest_memory: GuestMemoryMmap,
    next_kvm_slot: AtomicU32,
    /// Interrupts used by KvmVm's devices
    pub interrupts: Mutex<HashMap<u32, RoutingEntry>>,
    /// Allocator for VM resources
    pub resource_allocator: Mutex<ResourceAllocator>,
    /// MMIO bus
    pub mmio_bus: Arc<Bus>,
    /// The global KVM state (fd + capabilities).
    pub kvm: Kvm,
    /// Userfaultfd kept open for snapshot restore.
    pub uffd: Option<Uffd>,
    /// Handles to vCPU threads.
    pub vcpus_handles: Mutex<Vec<VcpuHandle>>,
    /// Event fd written to by vCPUs on exit.
    pub vcpus_exit_evt: EventFd,
    /// Set once an evicting snapshot has discarded guest memory, after which the guest cannot run.
    memory_evicted: AtomicBool,
    /// Test-only countdown that forces the Nth-next `set_user_memory_region` call to fail, used
    /// to exercise partial-failure handling. 0 means never fail.
    #[cfg(test)]
    fail_set_user_memory_region_in: AtomicU32,
}

/// Errors associated with the wrappers over KVM ioctls.
/// Needs `rustfmt::skip` to make multiline comments work
#[rustfmt::skip]
#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum VmError {
    /// Cannot set the memory regions: {0}
    SetUserMemoryRegion(kvm_ioctls::Error),
    /// Failed to create VM: {0}
    CreateVm(kvm_ioctls::Error),
    /// Failed to get KVM's dirty log: {0}
    GetDirtyLog(kvm_ioctls::Error),
    /// {0}
    Arch(#[from] KvmVmError),
    /// Error during eventfd operations: {0}
    EventFd(std::io::Error),
    /// Failed to create vcpu: {0}
    CreateVcpu(VcpuError),
    /// The number of configured slots is bigger than the maximum reported by KVM: {0}
    NotEnoughMemorySlots(u32),
    /// Failed to add a memory region: {0}
    InsertRegion(#[from] vm_memory::GuestRegionCollectionError),
    /// Failed to read /proc/self/pagemap: {0}
    PagemapRead(#[source] std::io::Error),
    /// ResourceAllocator error: {0}
    ResourceAllocator(#[from] vm_allocator::Error),
    /// MemoryError error: {0}
    MemoryError(#[from] MemoryError),
}

/// VM abstraction: either a KVM-based VM or a Nitro Enclave.
#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum Vm {
    /// KVM-backed virtual machine.
    Kvm(Arc<KvmVm>),
    /// Nitro Enclave-backed virtual machine.
    Enclave(EnclaveVm),
}

impl Vm {
    /// Returns the name of the VM type.
    pub fn type_name(&self) -> &'static str {
        match self {
            Vm::Kvm(_) => "Kvm",
            Vm::Enclave(_) => "Enclave",
        }
    }

    /// Returns a reference to the inner KVM VM, or `None` if this is not a KVM VM.
    pub fn as_kvm(&self) -> Option<&Arc<KvmVm>> {
        match self {
            Vm::Kvm(v) => Some(v),
            Vm::Enclave(_) => None,
        }
    }

    /// Returns a reference to the inner EnclaveVm, or `None` if this is not an Enclave VM.
    pub fn as_enclave(&self) -> Option<&EnclaveVm> {
        match self {
            Vm::Kvm(_) => None,
            Vm::Enclave(e) => Some(e),
        }
    }
}

/// Contains KvmVm functions that are usable across CPU architectures
impl KvmVm {
    /// Create a KVM VM
    pub fn create_common(kvm: Kvm) -> Result<VmCommon, VmError> {
        // It is known that KVM_CREATE_VM occasionally fails with EINTR on heavily loaded machines
        // with many VMs.
        //
        // The behavior itself that KVM_CREATE_VM can return EINTR is intentional. This is because
        // the KVM_CREATE_VM path includes mm_take_all_locks() that is CPU intensive and all CPU
        // intensive syscalls should check for pending signals and return EINTR immediately to allow
        // userland to remain interactive.
        // https://lists.nongnu.org/archive/html/qemu-devel/2014-01/msg01740.html
        //
        // However, it is empirically confirmed that, even though there is no pending signal,
        // KVM_CREATE_VM returns EINTR.
        // https://lore.kernel.org/qemu-devel/8735e0s1zw.wl-maz@kernel.org/
        //
        // To mitigate it, QEMU does an infinite retry on EINTR that greatly improves reliabiliy:
        // - https://github.com/qemu/qemu/commit/94ccff133820552a859c0fb95e33a539e0b90a75
        // - https://github.com/qemu/qemu/commit/bbde13cd14ad4eec18529ce0bf5876058464e124
        //
        // Similarly, we do retries up to 5 times. Although Firecracker clients are also able to
        // retry, they have to start Firecracker from scratch. Doing retries in Firecracker makes
        // recovery faster and improves reliability.
        const MAX_ATTEMPTS: u32 = 5;
        let mut attempt = 1;
        let fd = loop {
            match kvm.fd.create_vm() {
                Ok(fd) => break fd,
                Err(e) if e.errno() == libc::EINTR && attempt < MAX_ATTEMPTS => {
                    info!("Attempt #{attempt} of KVM_CREATE_VM returned EINTR");
                    // Exponential backoff (1us, 2us, 4us, and 8us => 15us in total)
                    std::thread::sleep(std::time::Duration::from_micros(2u64.pow(attempt - 1)));
                }
                Err(e) => return Err(VmError::CreateVm(e)),
            }

            attempt += 1;
        };

        let vcpus_exit_evt = EventFd::new(libc::EFD_NONBLOCK).map_err(VmError::EventFd)?;

        Ok(VmCommon {
            fd,
            max_memslots: kvm.max_nr_memslots(),
            guest_memory: GuestMemoryMmap::default(),
            next_kvm_slot: AtomicU32::new(0),
            interrupts: Mutex::new(HashMap::with_capacity(GSI_MSI_END as usize + 1)),
            resource_allocator: Mutex::new(ResourceAllocator::new()),
            mmio_bus: Arc::new(Bus::new()),
            kvm,
            uffd: None,
            vcpus_handles: Mutex::new(Vec::new()),
            vcpus_exit_evt,
            memory_evicted: AtomicBool::new(false),
            #[cfg(test)]
            fail_set_user_memory_region_in: AtomicU32::new(0),
        })
    }

    /// Creates the specified number of [`Vcpu`]s.
    ///
    /// Each vCPU gets a clone of the `vcpus_exit_evt` EventFd stored on this KvmVm.
    ///
    /// This also creates the irqchip, which KVM registers on its io bus. Since Linux 6.18 (commit
    /// 7d9a0273c459, "KVM: Avoid synchronize_srcu() in kvm_io_bus_register_dev()") that
    /// registration starts an SRCU grace period, and a memslot update issued right after it waits
    /// for the grace period to end (tens of ms), so register every memslot before calling this.
    /// Intel hosts still wait once: KVM adds its own APIC access page memslot while creating the
    /// first vCPU.
    pub fn create_vcpus(&self, vcpu_count: u8) -> Result<Vec<Vcpu>, VmError> {
        self.arch_pre_create_vcpus(vcpu_count)?;

        let mut vcpus = Vec::with_capacity(vcpu_count as usize);
        for cpu_idx in 0..vcpu_count {
            let exit_evt = self
                .vcpus_exit_evt()
                .try_clone()
                .map_err(VmError::EventFd)?;
            let vcpu = Vcpu::new(cpu_idx, self, exit_evt).map_err(VmError::CreateVcpu)?;
            vcpus.push(vcpu);
        }

        self.arch_post_create_vcpus(vcpu_count)?;

        Ok(vcpus)
    }

    /// Returns a reference to the [`Kvm`] instance.
    pub fn kvm(&self) -> &Kvm {
        &self.common.kvm
    }

    /// Returns a reference to the vCPU exit [`EventFd`].
    pub fn vcpus_exit_evt(&self) -> &EventFd {
        &self.common.vcpus_exit_evt
    }

    /// Returns a locked reference to the vCPU handles.
    pub fn vcpus_handles(&self) -> MutexGuard<'_, Vec<VcpuHandle>> {
        self.common.vcpus_handles.lock().expect("Poisoned lock")
    }

    /// Sets the userfaultfd (used during snapshot restore).
    pub fn set_uffd(&mut self, uffd: Option<Uffd>) {
        self.common.uffd = uffd;
    }

    /// Starts the microVM vCPUs.
    ///
    /// Sets the terminal to raw/non-blocking mode, then spawns a thread per vCPU
    /// and stores the resulting handles. The barrier is used to synchronize TLS
    /// initialization across all vCPU threads before returning.
    pub fn start_vcpus(
        self: &Arc<Self>,
        mut vcpus: Vec<Vcpu>,
        vcpu_seccomp_filter: Arc<crate::seccomp::BpfProgram>,
    ) -> Result<(), StartVcpusError> {
        let vcpu_count = vcpus.len();
        let barrier = Arc::new(Barrier::new(vcpu_count + 1));

        let stdin = std::io::stdin().lock();
        stdin.set_raw_mode().inspect_err(|&err| {
            crate::logger::warn!("Cannot set raw mode for the terminal. {:?}", err);
        })?;
        stdin.set_non_block(true).inspect_err(|&err| {
            crate::logger::warn!("Cannot set non block for the terminal. {:?}", err);
        })?;

        let mut handles = self.vcpus_handles();
        handles.reserve(vcpu_count);
        for mut vcpu in vcpus.drain(..) {
            vcpu.set_mmio_bus(self.common.mmio_bus.clone());
            #[cfg(target_arch = "x86_64")]
            vcpu.kvm_vcpu.set_pio_bus(self.pio_bus.clone());

            handles.push(vcpu.start_threaded(
                self,
                vcpu_seccomp_filter.clone(),
                barrier.clone(),
            )?);
        }
        drop(handles);
        barrier.wait();

        Ok(())
    }

    /// Sends a pause event to all vCPUs and waits for acknowledgement.
    pub fn pause_vcpus(&self) -> Result<(), crate::VmmError> {
        let mut handles = self.vcpus_handles();
        handles
            .iter_mut()
            .try_for_each(|handle| handle.send_event(crate::VcpuEvent::Pause))
            .map_err(|_| crate::VmmError::VcpuMessage)?;

        if handles
            .iter()
            .map(|handle| {
                handle
                    .response_receiver()
                    .recv_timeout(crate::RECV_TIMEOUT_SEC)
            })
            .any(|response| !matches!(response, Ok(crate::VcpuResponse::Paused)))
        {
            return Err(crate::VmmError::VcpuMessage);
        }
        Ok(())
    }

    /// Sends a resume event to all vCPUs and waits for acknowledgement.
    pub fn resume_vcpus(&self) -> Result<(), crate::VmmError> {
        let mut handles = self.vcpus_handles();
        handles
            .iter_mut()
            .try_for_each(|handle| handle.send_event(crate::VcpuEvent::Resume))
            .map_err(|_| crate::VmmError::VcpuMessage)?;

        if handles
            .iter()
            .map(|handle| {
                handle
                    .response_receiver()
                    .recv_timeout(crate::RECV_TIMEOUT_SEC)
            })
            .any(|response| !matches!(response, Ok(crate::VcpuResponse::Resumed)))
        {
            return Err(crate::VmmError::VcpuMessage);
        }
        Ok(())
    }

    /// Saves vCPU states by requesting each vCPU thread to serialize its state.
    pub fn save_vcpu_states(
        &self,
    ) -> Result<Vec<crate::vstate::vcpu::VcpuState>, crate::persist::MicrovmStateError> {
        use crate::persist::MicrovmStateError;

        let mut handles = self.vcpus_handles();
        for handle in handles.iter_mut() {
            handle
                .send_event(crate::VcpuEvent::SaveState)
                .map_err(MicrovmStateError::SignalVcpu)?;
        }

        let vcpu_responses = handles
            .iter()
            .map(|handle| {
                handle
                    .response_receiver()
                    .recv_timeout(crate::RECV_TIMEOUT_SEC)
            })
            .collect::<Result<Vec<crate::VcpuResponse>, _>>()
            .map_err(|_| MicrovmStateError::UnexpectedVcpuResponse)?;

        vcpu_responses
            .into_iter()
            .map(|response| match response {
                crate::VcpuResponse::SavedState(state) => Ok(*state),
                crate::VcpuResponse::Error(err) => Err(MicrovmStateError::SaveVcpuState(err)),
                crate::VcpuResponse::NotAllowed(reason) => {
                    Err(MicrovmStateError::NotAllowed(reason))
                }
                _ => Err(MicrovmStateError::UnexpectedVcpuResponse),
            })
            .collect()
    }

    /// Dumps CPU configuration from all vCPU threads.
    pub fn dump_cpu_config_states(
        &self,
    ) -> Result<Vec<crate::cpu_config::templates::CpuConfiguration>, crate::DumpCpuConfigError>
    {
        use crate::DumpCpuConfigError;

        let mut handles = self.vcpus_handles();
        for handle in handles.iter_mut() {
            handle
                .send_event(crate::VcpuEvent::DumpCpuConfig)
                .map_err(DumpCpuConfigError::SendEvent)?;
        }

        let vcpu_responses = handles
            .iter()
            .map(|handle| {
                handle
                    .response_receiver()
                    .recv_timeout(crate::RECV_TIMEOUT_SEC)
            })
            .collect::<Result<Vec<crate::VcpuResponse>, _>>()
            .map_err(|_| DumpCpuConfigError::UnexpectedResponse)?;

        vcpu_responses
            .into_iter()
            .map(|response| match response {
                crate::VcpuResponse::DumpedCpuConfig(cpu_config) => Ok(*cpu_config),
                crate::VcpuResponse::Error(err) => Err(DumpCpuConfigError::DumpCpuConfig(err)),
                crate::VcpuResponse::NotAllowed(reason) => {
                    Err(DumpCpuConfigError::NotAllowed(reason))
                }
                _ => Err(DumpCpuConfigError::UnexpectedResponse),
            })
            .collect()
    }

    /// Sends finish events to all vCPU threads and joins them.
    pub fn shutdown_vcpus(&self) {
        let mut handles = self.vcpus_handles();
        for (idx, handle) in handles.iter_mut().enumerate() {
            if let Err(err) = handle.send_event(crate::VcpuEvent::Finish) {
                crate::logger::error!("Failed to send VcpuEvent::Finish to vCPU {}: {}", idx, err);
            }
        }
        // Join the vCPU threads by running VcpuHandle::drop().
        handles.clear();
    }

    /// Reserves the next `slot_cnt` contiguous kvm slot ids and returns the first one
    pub fn next_kvm_slot(&self, slot_cnt: u32) -> Option<u32> {
        let next = self
            .common
            .next_kvm_slot
            .fetch_add(slot_cnt, Ordering::Relaxed);
        if self.common.max_memslots <= next {
            None
        } else {
            Some(next)
        }
    }

    pub(crate) fn set_user_memory_region(
        &self,
        region: kvm_userspace_memory_region,
    ) -> Result<(), VmError> {
        #[cfg(test)]
        if self.common.fail_set_user_memory_region_in.fetch_update(
            Ordering::Relaxed,
            Ordering::Relaxed,
            |n| (n > 0).then(|| n - 1),
        ) == Ok(1)
        {
            return Err(VmError::SetUserMemoryRegion(kvm_ioctls::Error::new(
                libc::ENOMEM,
            )));
        }
        // SAFETY: Safe because the fd is a valid KVM file descriptor.
        unsafe {
            self.fd()
                .set_user_memory_region(region)
                .map_err(VmError::SetUserMemoryRegion)
        }
    }

    /// Test-only: arm the fault injector so the `n`th subsequent call to
    /// [`Self::set_user_memory_region`] fails (n == 1 fails the very next call).
    #[cfg(test)]
    pub(crate) fn fail_next_set_user_memory_region(&self, n: u32) {
        self.common
            .fail_set_user_memory_region_in
            .store(n, Ordering::Relaxed);
    }

    fn register_memory_region(&mut self, region: Arc<GuestRegionMmapExt>) -> Result<(), VmError> {
        let new_guest_memory = self
            .common
            .guest_memory
            .insert_region(Arc::clone(&region))?;

        region
            .slots()
            .try_for_each(|(ref slot, plugged)| match plugged {
                // if the slot is plugged, add it to kvm user memory regions
                true => self.set_user_memory_region(slot.into()),
                // if the slot is not plugged, protect accesses to it
                false => slot.protect(true).map_err(VmError::MemoryError),
            })?;

        self.common.guest_memory = new_guest_memory;

        Ok(())
    }

    /// Register a list of new memory regions to this [`KvmVm`].
    pub fn register_dram_memory_regions(
        &mut self,
        regions: Vec<GuestRegionMmap>,
    ) -> Result<(), VmError> {
        for region in regions {
            let next_slot = self
                .next_kvm_slot(1)
                .ok_or(VmError::NotEnoughMemorySlots(self.common.max_memslots))?;

            let arcd_region =
                Arc::new(GuestRegionMmapExt::dram_from_mmap_region(region, next_slot));

            self.register_memory_region(arcd_region)?
        }

        Ok(())
    }

    /// Register a new hotpluggable region to this [`KvmVm`].
    pub fn register_hotpluggable_memory_region(
        &mut self,
        region: GuestRegionMmap,
        slot_size: usize,
    ) -> Result<(), VmError> {
        // caller should ensure the slot size divides the region length.
        assert!(region.len().is_multiple_of(slot_size as u64));
        let slot_cnt = (region.len() / (slot_size as u64))
            .try_into()
            .map_err(|_| VmError::NotEnoughMemorySlots(self.common.max_memslots))?;
        let slot_from = self
            .next_kvm_slot(slot_cnt)
            .ok_or(VmError::NotEnoughMemorySlots(self.common.max_memslots))?;
        let arcd_region = Arc::new(GuestRegionMmapExt::hotpluggable_from_mmap_region(
            region, slot_from, slot_size,
        ));

        self.register_memory_region(arcd_region)
    }

    /// Register a list of new memory regions to this [`KvmVm`].
    ///
    /// Note: regions and state.regions need to be in the same order.
    pub fn restore_memory_regions(
        &mut self,
        regions: Vec<GuestRegionMmap>,
        state: &GuestMemoryState,
    ) -> Result<(), VmError> {
        for (region, state) in regions.into_iter().zip(state.regions.iter()) {
            let slot_cnt = state
                .plugged
                .len()
                .try_into()
                .map_err(|_| VmError::NotEnoughMemorySlots(self.common.max_memslots))?;

            let next_slot = self
                .next_kvm_slot(slot_cnt)
                .ok_or(VmError::NotEnoughMemorySlots(self.common.max_memslots))?;

            let arcd_region = Arc::new(GuestRegionMmapExt::from_state(region, state, next_slot)?);

            self.register_memory_region(arcd_region)?
        }

        Ok(())
    }

    /// Gets a reference to the kvm file descriptor owned by this VM.
    pub fn fd(&self) -> &VmFd {
        &self.common.fd
    }

    /// Gets a reference to this [`KvmVm`]'s [`GuestMemoryMmap`] object
    pub fn guest_memory(&self) -> &GuestMemoryMmap {
        &self.common.guest_memory
    }

    /// Gets a mutable reference to this [`KvmVm`]'s [`ResourceAllocator`] object
    pub fn resource_allocator(&self) -> MutexGuard<'_, ResourceAllocator> {
        self.common
            .resource_allocator
            .lock()
            .expect("Poisoned lock")
    }

    /// Resets the KVM dirty bitmap for each of the guest's memory regions.
    pub fn reset_dirty_bitmap(&self) {
        self.guest_memory()
            .iter()
            .flat_map(|region| region.plugged_slots())
            .for_each(|mem_slot| {
                let _ = self.fd().get_dirty_log(mem_slot.slot, mem_slot.slice.len());
            });
    }

    /// Retrieves the KVM dirty bitmap for each of the guest's memory regions.
    ///
    /// A slot without dirty tracking takes its page set from `/proc/self/pagemap`, which this
    /// opens once for all of the slots.
    pub fn get_dirty_bitmap(&self) -> Result<DirtyBitmap, VmError> {
        let mut pagemap: Option<File> = None;
        self.guest_memory()
            .iter()
            .flat_map(|region| region.plugged_slots())
            .map(|mem_slot| {
                let bitmap = match mem_slot.slice.bitmap() {
                    Some(_) => self
                        .fd()
                        .get_dirty_log(mem_slot.slot, mem_slot.slice.len())
                        .map_err(VmError::GetDirtyLog)?,
                    None => {
                        if pagemap.is_none() {
                            pagemap = Some(File::open(PAGEMAP_PATH).map_err(VmError::PagemapRead)?);
                        }
                        pagemap_bitmap(
                            pagemap.as_mut().expect("opened above"),
                            mem_slot.slice.ptr_guard_mut().as_ptr(),
                            mem_slot.slice.len(),
                        )?
                    }
                };
                Ok((mem_slot.slot, bitmap))
            })
            .collect()
    }

    /// Takes a snapshot of the virtual machine running inside the given [`Vmm`] and saves it to
    /// `mem_file_path`.
    ///
    /// If `snapshot_type` is [`SnapshotType::Diff`], and `mem_file_path` exists and is a snapshot
    /// file of matching size, then the diff snapshot will be directly merged into the existing
    /// snapshot. Otherwise, existing files are simply overwritten.
    ///
    /// With `evict`, guest memory is written without the host page cache and discarded chunk by
    /// chunk as it is written, so the guest can no longer run; see [`KvmVm::memory_evicted`].
    pub(crate) fn snapshot_memory_to_file(
        &self,
        mem_file_path: &Path,
        snapshot_type: SnapshotType,
        sync_snapshot_files: bool,
        evict: bool,
    ) -> Result<(), CreateSnapshotError> {
        use self::CreateSnapshotError::*;

        // Need to check this here, as we create the file in the line below
        let file_existed = mem_file_path.exists();
        let mut merges_into_file = false;

        let mut options = OpenOptions::new();
        options.write(true).create(true).truncate(false);
        let (mut file, evict_write) = if evict {
            let (file, mode) = open_evicting(|direct| {
                let mut options = options.clone();
                if direct {
                    options.custom_flags(libc::O_DIRECT);
                }
                options.open(mem_file_path)
            })
            .map_err(|err| MemoryBackingFile("open", err))?;
            match mode {
                EvictWrite::Direct => {
                    info!("Evicting snapshot: writing the memory file with O_DIRECT")
                }
                EvictWrite::Buffered => warn!(
                    "Evicting snapshot: the file system refused O_DIRECT, so the memory file is \
                     written through the page cache and each chunk is dropped from it"
                ),
            }
            (file, Some(mode))
        } else {
            let file = options
                .open(mem_file_path)
                .map_err(|err| MemoryBackingFile("open", err))?;
            (file, None)
        };

        // Determine what size our total memory area is.
        let mem_size_mib = mem_size_mib(self.guest_memory());
        let expected_size = mem_size_mib * 1024 * 1024;

        if file_existed {
            let file_size = file
                .metadata()
                .map_err(|e| MemoryBackingFile("get_metadata", e))?
                .len();

            // Here we only truncate the file if the size mismatches.
            // - For full snapshots, the entire file's contents will be overwritten anyway. We have
            //   to avoid truncating here to deal with the edge case where it represents the
            //   snapshot file from which this very microVM was loaded (as modifying the memory file
            //   would be reflected in the mmap of the file, meaning a truncate operation would zero
            //   out guest memory, and thus corrupt the VM).
            // - For diff snapshots, we want to merge the diff layer directly into the file.
            if file_size != expected_size {
                file.set_len(0)
                    .map_err(|err| MemoryBackingFile("truncate", err))?;
            } else {
                merges_into_file = true;
            }
        }

        // Set the length of the file to the full size of the memory area.
        file.set_len(expected_size)
            .map_err(|e| MemoryBackingFile("set_length", e))?;

        match (snapshot_type, evict_write) {
            (_, Some(mode)) => self.dump_evicting(&file, snapshot_type, mode, merges_into_file)?,
            (SnapshotType::Diff, None) => {
                let dirty_bitmap = self.get_dirty_bitmap()?;
                // A diff merged into the full memory file punches a hole for each dirty page that
                // reads as zero, such as a page the guest freed since the last snapshot, so the
                // file frees that disk space instead of storing the zeros. A separate diff layer
                // keeps writing them, since a hole there marks a page the layer does not hold.
                self.guest_memory()
                    .dump_dirty(&mut file, &dirty_bitmap, merges_into_file)?;
            }
            (SnapshotType::Full, None) => {
                self.guest_memory().dump(&mut file)?;
                self.reset_dirty_bitmap();
                self.guest_memory().reset_dirty();
            }
        };

        file.flush()
            .map_err(|err| MemoryBackingFile("flush", err))?;
        if sync_snapshot_files {
            file.sync_all()
                .map_err(|err| MemoryBackingFile("sync_all", err))?;
        }
        Ok(())
    }

    /// Dumps guest memory into `file` and discards it as it goes, then marks the guest's memory
    /// evicted. A dump that fails after its first discard marks it evicted too, since the guest's
    /// memory is no longer whole.
    fn dump_evicting(
        &self,
        file: &File,
        snapshot_type: SnapshotType,
        mode: EvictWrite,
        merges_into_file: bool,
    ) -> Result<(), CreateSnapshotError> {
        let mut discarded = false;
        let dumped = match snapshot_type {
            // A diff onto a copy of the previous memory file punches each dirty run that reads
            // as zero, so the file keeps none of the previous file's bytes there.
            SnapshotType::Diff => self
                .get_dirty_bitmap()
                .map_err(Into::into)
                .and_then(|bitmap| {
                    self.guest_memory()
                        .dump_dirty_evicting(file, mode, &bitmap, merges_into_file, &mut discarded)
                        .map_err(Into::into)
                }),
            SnapshotType::Full => self
                .guest_memory()
                .dump_evicting(file, mode, &mut discarded)
                .map(|()| {
                    self.reset_dirty_bitmap();
                    self.guest_memory().reset_dirty();
                })
                .map_err(Into::into),
        };
        if discarded || dumped.is_ok() {
            self.common.memory_evicted.store(true, Ordering::Release);
        }
        dumped
    }

    /// Whether an evicting snapshot has discarded guest memory, so the guest cannot run again.
    pub fn memory_evicted(&self) -> bool {
        self.common.memory_evicted.load(Ordering::Acquire)
    }

    /// Register a device IRQ
    pub fn register_irq(&self, fd: &EventFd, gsi: u32) -> Result<(), errno::Error> {
        self.common.fd.register_irqfd(fd, gsi)?;

        let mut entry = kvm_irq_routing_entry {
            gsi,
            type_: KVM_IRQ_ROUTING_IRQCHIP,
            ..Default::default()
        };
        #[cfg(target_arch = "x86_64")]
        {
            entry.u.irqchip.irqchip = KVM_IRQCHIP_IOAPIC;
        }
        #[cfg(target_arch = "aarch64")]
        {
            entry.u.irqchip.irqchip = 0;
        }
        entry.u.irqchip.pin = gsi;

        self.common
            .interrupts
            .lock()
            .expect("Poisoned lock")
            .insert(
                gsi,
                RoutingEntry {
                    entry,
                    masked: false,
                },
            );
        Ok(())
    }

    /// Register an MSI device interrupt
    pub fn register_msi(
        &self,
        route: &MsixVector,
        table_entry: &MsixTableEntry,
        pci_sbdf: PciSBDF,
    ) -> Result<(), errno::Error> {
        let mut entry = kvm_irq_routing_entry {
            gsi: route.gsi,
            type_: KVM_IRQ_ROUTING_MSI,
            ..Default::default()
        };
        entry.u.msi.address_lo = table_entry.msg_addr_lo;
        entry.u.msi.address_hi = table_entry.msg_addr_hi;
        entry.u.msi.data = table_entry.msg_data;

        if self.common.fd.check_extension(kvm_ioctls::Cap::MsiDevid) {
            entry.flags = KVM_MSI_VALID_DEVID;
            entry.u.msi.__bindgen_anon_1.devid = pci_sbdf.into();
        }

        self.common
            .interrupts
            .lock()
            .expect("Poisoned lock")
            .insert(
                route.gsi,
                RoutingEntry {
                    entry,
                    masked: table_entry.masked(),
                },
            );

        Ok(())
    }

    /// Create a group of MSI-X interrupts
    pub fn create_msix_group(
        vm: Arc<KvmVm>,
        count: u16,
    ) -> Result<MsixVectorGroup, InterruptError> {
        debug!("Creating new MSI group with {count} vectors");
        let mut vectors = Vec::with_capacity(count as usize);
        for gsi in vm
            .resource_allocator()
            .allocate_gsi_msi(count as u32)?
            .iter()
        {
            vectors.push(MsixVector::new(*gsi, false)?);
        }

        Ok(MsixVectorGroup { vm, vectors })
    }

    /// Set GSI routes to KVM
    pub fn set_gsi_routes(&self) -> Result<(), InterruptError> {
        let entries = self.common.interrupts.lock().expect("Poisoned lock");
        let unmasked = entries.values().filter(|e| !e.masked).count();
        let mut routes = KvmIrqRouting::new(unmasked)?;
        for (slot, entry) in entries.values().filter(|e| !e.masked).enumerate() {
            routes.as_mut_slice()[slot] = entry.entry;
        }

        self.common.fd.set_gsi_routing(&routes)?;
        Ok(())
    }
}

/// Opens the memory file of an evicting snapshot through `open`, which takes whether to add
/// `O_DIRECT`. A file system that refuses `O_DIRECT` with `EINVAL`, as tmpfs did before Linux
/// 6.6, gets the file opened without it.
fn open_evicting(
    open: impl Fn(bool) -> std::io::Result<File>,
) -> std::io::Result<(File, EvictWrite)> {
    match open(true) {
        Ok(file) => Ok((file, EvictWrite::Direct)),
        Err(err) if err.raw_os_error() == Some(libc::EINVAL) => {
            Ok((open(false)?, EvictWrite::Buffered))
        }
        Err(err) => Err(err),
    }
}

/// The file the VMM reads to tell the pages the guest wrote from the pages it only read.
const PAGEMAP_PATH: &str = "/proc/self/pagemap";
/// A `/proc/<pid>/pagemap` entry: the page is present in RAM.
const PAGEMAP_PRESENT: u64 = 1 << 63;
/// A `/proc/<pid>/pagemap` entry: the page is in swap.
const PAGEMAP_SWAPPED: u64 = 1 << 62;
/// A `/proc/<pid>/pagemap` entry: the page is a file page or shared anonymous.
const PAGEMAP_FILE_OR_SHARED: u64 = 1 << 61;
/// How many `pagemap` entries one `read` fetches.
const PAGEMAP_CHUNK_ENTRIES: usize = 8192;

/// Whether the page that a `/proc/<pid>/pagemap` entry describes holds data the guest wrote.
///
/// A page of a private mapping that the guest wrote is an anonymous copy, either present or in
/// swap. A page it only read stays a page of the file, which has bit 61 set. An absent page, with
/// no bit set, holds nothing the guest wrote.
fn pagemap_entry_written(entry: u64) -> bool {
    entry & PAGEMAP_SWAPPED != 0
        || (entry & PAGEMAP_PRESENT != 0 && entry & PAGEMAP_FILE_OR_SHARED == 0)
}

/// Reads the pages the guest wrote in the memslot at `addr`, `len` bytes long, from the
/// `/proc/self/pagemap` file `pagemap`, as a bitmap with one bit for each page. To be used if a
/// diff snapshot is requested, but dirty page tracking wasn't enabled.
///
/// The set holds every page written since the memory was mapped, swapped out ones included, and
/// no page that was only read from the file backing a private mapping. It does not reset after a
/// snapshot, as KVM's dirty log does.
fn pagemap_bitmap(pagemap: &mut File, addr: *mut u8, len: usize) -> Result<Vec<u64>, VmError> {
    let page_size = host_page_size();
    let pages = len / page_size;
    let mut bitmap = vec![0u64; pages.div_ceil(64)];

    // The file holds one 8 byte entry for each page of the address space.
    let offset = (addr as usize / page_size) as u64 * 8;
    pagemap
        .seek(SeekFrom::Start(offset))
        .map_err(VmError::PagemapRead)?;

    let mut entries = vec![0u8; PAGEMAP_CHUNK_ENTRIES.min(pages) * 8];
    let mut page_idx = 0;
    while page_idx < pages {
        let chunk = &mut entries[..PAGEMAP_CHUNK_ENTRIES.min(pages - page_idx) * 8];
        pagemap.read_exact(chunk).map_err(VmError::PagemapRead)?;
        for entry in chunk.chunks_exact(8) {
            let entry = u64::from_ne_bytes(entry.try_into().expect("chunks of 8 bytes"));
            bitmap[page_idx / 64] |= u64::from(pagemap_entry_written(entry)) << (page_idx % 64);
            page_idx += 1;
        }
    }

    Ok(bitmap)
}

#[cfg(test)]
pub(crate) mod tests {
    use std::assert_matches;
    use std::os::fd::AsRawFd;
    use std::sync::atomic::Ordering;

    use vm_memory::GuestAddress;
    use vmm_sys_util::tempfile::TempFile;

    use super::*;
    use crate::arch;
    use crate::pci::PciSBDF;
    use crate::pci::msix::MsixTableEntry;
    use crate::snapshot::Persist;
    use crate::test_utils::single_region_mem_raw;
    use crate::utils::mib_to_bytes;
    use crate::vmm_config::machine_config::HugePageConfig;
    use crate::vstate::kvm::Kvm;
    use crate::vstate::memory::anonymous;

    // Auxiliary function being used throughout the tests.
    pub(crate) fn setup_vm() -> KvmVm {
        let kvm = Kvm::new(vec![]).expect("Cannot create Kvm");
        KvmVm::new(kvm).expect("Cannot create new vm")
    }

    // Auxiliary function being used throughout the tests.
    pub(crate) fn setup_vm_with_memory(mem_size: usize) -> KvmVm {
        let mut vm = setup_vm();
        let gm = single_region_mem_raw(mem_size);
        vm.register_dram_memory_regions(gm).unwrap();
        vm
    }

    #[test]
    fn test_open_evicting_falls_back_without_o_direct() {
        let temp = vmm_sys_util::tempfile::TempFile::new().unwrap();
        // Opens the file, failing with `refuse` when asked for O_DIRECT.
        let open = |refuse: Option<i32>| {
            let path = temp.as_path().to_path_buf();
            move |direct: bool| match refuse {
                Some(errno) if direct => Err(std::io::Error::from_raw_os_error(errno)),
                _ => OpenOptions::new().write(true).open(&path),
            }
        };
        assert_eq!(open_evicting(open(None)).unwrap().1, EvictWrite::Direct);
        // A file system without O_DIRECT gets a buffered file.
        assert_eq!(
            open_evicting(open(Some(libc::EINVAL))).unwrap().1,
            EvictWrite::Buffered
        );
        // Any other error is returned.
        assert_eq!(
            open_evicting(open(Some(libc::EACCES)))
                .unwrap_err()
                .raw_os_error(),
            Some(libc::EACCES)
        );
    }

    #[test]
    fn test_snapshot_memory_to_file_evict() {
        use vm_memory::Bytes;

        let mem_size = mib_to_bytes(4);
        for snapshot_type in [SnapshotType::Full, SnapshotType::Diff] {
            for evict in [false, true] {
                let vm = setup_vm_with_memory(mem_size);
                vm.guest_memory()
                    .write(&vec![5u8; mem_size], GuestAddress(0))
                    .unwrap();
                let memory_file = vmm_sys_util::tempfile::TempFile::new().unwrap();
                vm.snapshot_memory_to_file(memory_file.as_path(), snapshot_type, false, evict)
                    .unwrap();

                // Both write the guest's memory; only the evicting snapshot discards it.
                assert_eq!(
                    std::fs::read(memory_file.as_path()).unwrap(),
                    vec![5u8; mem_size]
                );
                assert_eq!(vm.memory_evicted(), evict);
                let mut contents = vec![0u8; mem_size];
                vm.guest_memory()
                    .read(&mut contents, GuestAddress(0))
                    .unwrap();
                let expected = if evict { 0 } else { 5 };
                assert!(contents.iter().all(|&byte| byte == expected));
            }
        }
    }

    #[test]
    fn test_new() {
        // Testing with a valid /dev/kvm descriptor.
        let kvm = Kvm::new(vec![]).expect("Cannot create Kvm");
        KvmVm::new(kvm).unwrap();
    }

    #[test]
    fn test_register_memory_regions() {
        let mut vm = setup_vm();

        // Trying to set a memory region with a size that is not a multiple of GUEST_PAGE_SIZE
        // will result in error.
        let gm = single_region_mem_raw(0x10);
        let res = vm.register_dram_memory_regions(gm);
        assert_eq!(
            res.unwrap_err().to_string(),
            "Cannot set the memory regions: Invalid argument (os error 22)"
        );

        let gm = single_region_mem_raw(0x1000);
        let res = vm.register_dram_memory_regions(gm);
        res.unwrap();
    }

    #[test]
    fn test_too_many_regions() {
        let mut vm = setup_vm();
        let max_nr_regions = vm.kvm().max_nr_memslots();

        for i in 0..=max_nr_regions {
            let regions = anonymous(
                vec![(GuestAddress(i as u64 * 0x1000), 0x1000)].into_iter(),
                false,
                HugePageConfig::None,
            )
            .unwrap();
            assert_eq!(regions.len(), 1);

            let res = vm.register_dram_memory_regions(regions);

            if max_nr_regions <= i {
                assert!(
                    matches!(res, Err(VmError::NotEnoughMemorySlots(v)) if v == max_nr_regions),
                    "{:?} at iteration {}",
                    res,
                    i
                );
            } else {
                res.unwrap_or_else(|_| {
                    panic!(
                        "to be able to insert more regions in iteration {i} - max_nr_memslots: \
                         {max_nr_regions} - num_regions: {}",
                        vm.guest_memory().num_regions()
                    )
                });
            }
        }
    }

    #[test]
    fn test_create_vcpus() {
        let vcpu_count = 2;
        let vm = setup_vm_with_memory(mib_to_bytes(128));

        let vcpu_vec = vm.create_vcpus(vcpu_count).unwrap();

        assert_eq!(vcpu_vec.len(), vcpu_count as usize);
    }

    fn enable_irqchip(vm: &KvmVm) {
        #[cfg(target_arch = "x86_64")]
        vm.setup_irqchip().unwrap();
        #[cfg(target_arch = "aarch64")]
        vm.setup_irqchip(1).unwrap();
    }

    #[test]
    fn test_msi_vector_group_new() {
        let vm = setup_vm_with_memory(mib_to_bytes(128));
        let vm = Arc::new(vm);
        let msix_group = KvmVm::create_msix_group(vm.clone(), 4).unwrap();
        assert_eq!(msix_group.num_vectors(), 4);
    }

    /// Enable all vectors MSI-X vector group
    pub fn enable_all_vectors(vector_group: &MsixVectorGroup) {
        for route in &vector_group.vectors {
            route.enable(&vector_group.vm.common.fd).unwrap();
        }
    }

    #[test]
    fn test_msi_vector_group_disable() {
        let vm = setup_vm_with_memory(mib_to_bytes(128));
        enable_irqchip(&vm);
        let vm = Arc::new(vm);
        let msix_group = KvmVm::create_msix_group(vm.clone(), 4).unwrap();

        // Initially all vectors are disabled
        for route in &msix_group.vectors {
            assert!(!route.enabled.load(Ordering::Acquire))
        }
        // Enabling vectors
        enable_all_vectors(&msix_group);

        // Disable works
        msix_group.disable().unwrap();
        for route in &msix_group.vectors {
            assert!(!route.enabled.load(Ordering::Acquire))
        }
        // Disabling a disabled group doesn't error out
    }

    #[test]
    fn test_msi_vector_group_trigger() {
        let vm = setup_vm_with_memory(mib_to_bytes(128));
        enable_irqchip(&vm);

        let vm = Arc::new(vm);
        let msix_group = KvmVm::create_msix_group(vm.clone(), 4).unwrap();

        // We can now trigger all vectors
        for i in 0..4 {
            msix_group.trigger(i).unwrap()
        }

        // We can't trigger an invalid vector
        msix_group.trigger(4).unwrap_err();
    }

    #[test]
    fn test_msi_vector_group_notifier() {
        let vm = setup_vm_with_memory(mib_to_bytes(128));
        let vm = Arc::new(vm);
        let msix_group = KvmVm::create_msix_group(vm.clone(), 4).unwrap();

        for i in 0..4 {
            assert!(msix_group.notifier(i).is_some());
        }

        assert!(msix_group.notifier(4).is_none());
    }

    #[test]
    fn test_msi_vector_group_update_invalid_vector() {
        let vm = setup_vm_with_memory(mib_to_bytes(128));
        enable_irqchip(&vm);
        let vm = Arc::new(vm);
        let msix_group = KvmVm::create_msix_group(vm.clone(), 4).unwrap();
        let sbdf = PciSBDF::from(0xafa);
        let config = MsixTableEntry {
            msg_addr_hi: 0x42,
            msg_addr_lo: 0x12,
            msg_data: 0x12,
            // Masked
            vector_ctl: 0x1,
        };
        msix_group.update(0, &config, sbdf).unwrap();
        assert_matches!(
            msix_group.update(4, &config, sbdf),
            Err(InterruptError::InvalidVectorIndex(4)),
        );
    }

    #[test]
    fn test_msi_vector_group_update() {
        let vm = setup_vm_with_memory(mib_to_bytes(128));
        enable_irqchip(&vm);
        let vm = Arc::new(vm);
        assert!(vm.common.interrupts.lock().unwrap().is_empty());
        let msix_group = KvmVm::create_msix_group(vm.clone(), 4).unwrap();
        let pci_sbdf = PciSBDF::from(0xafa);

        // Set some configuration for the vectors. Initially all are masked
        let configs: [_; 4] = std::array::from_fn(|idx| MsixTableEntry {
            msg_addr_hi: 0x42,
            msg_addr_lo: 0x13,
            msg_data: 0x12 * u32::try_from(idx).unwrap(),
            vector_ctl: 0x1, // Masked
        });
        msix_group.update_batched(&configs, pci_sbdf).unwrap();

        // All vectors should be disabled
        for vector in &msix_group.vectors {
            assert!(!vector.enabled.load(Ordering::Acquire));
        }

        for i in 0..4 {
            let gsi = crate::arch::GSI_MSI_START + i;
            let interrupts = vm.common.interrupts.lock().unwrap();
            let kvm_route = interrupts.get(&gsi).unwrap();
            assert!(kvm_route.masked);
            assert_eq!(kvm_route.entry.gsi, gsi);
            assert_eq!(kvm_route.entry.type_, KVM_IRQ_ROUTING_MSI);
            // SAFETY: because we know we setup MSI routes.
            unsafe {
                assert_eq!(kvm_route.entry.u.msi.address_hi, 0x42);
                assert_eq!(kvm_route.entry.u.msi.address_lo, 0x13);
                assert_eq!(kvm_route.entry.u.msi.data, 0x12 * i);
            }
        }

        // Simply enabling the vectors should not update the registered IRQ routes
        enable_all_vectors(&msix_group);
        for i in 0..4 {
            let gsi = crate::arch::GSI_MSI_START + i;
            let interrupts = vm.common.interrupts.lock().unwrap();
            let kvm_route = interrupts.get(&gsi).unwrap();
            assert!(kvm_route.masked);
            assert_eq!(kvm_route.entry.gsi, gsi);
            assert_eq!(kvm_route.entry.type_, KVM_IRQ_ROUTING_MSI);
            // SAFETY: because we know we setup MSI routes.
            unsafe {
                assert_eq!(kvm_route.entry.u.msi.address_hi, 0x42);
                assert_eq!(kvm_route.entry.u.msi.address_lo, 0x13);
                assert_eq!(kvm_route.entry.u.msi.data, 0x12 * i);
            }
            assert!(
                msix_group.vectors[i as usize]
                    .enabled
                    .load(Ordering::Acquire)
            );
        }
        msix_group.disable().unwrap();

        let unmasked0 = MsixTableEntry {
            vector_ctl: 0x0, // un-masked
            ..configs[0].clone()
        };
        let unmasked1 = MsixTableEntry {
            vector_ctl: 0x0, // un-masked
            ..configs[1].clone()
        };
        // `update` and `register` will update the in-memory GSI config of its route (and only its route)
        msix_group.update(0, &unmasked0, pci_sbdf).unwrap();
        msix_group.register(1, &unmasked1, pci_sbdf).unwrap();
        for i in 0..4 {
            let gsi = crate::arch::GSI_MSI_START + i;
            let interrupts = vm.common.interrupts.lock().unwrap();
            let kvm_route = interrupts.get(&gsi).unwrap();
            assert_eq!(kvm_route.masked, i >= 2);
            assert_eq!(kvm_route.entry.gsi, gsi);
            assert_eq!(kvm_route.entry.type_, KVM_IRQ_ROUTING_MSI);
            // SAFETY: because we know we setup MSI routes.
            unsafe {
                assert_eq!(kvm_route.entry.u.msi.address_hi, 0x42);
                assert_eq!(kvm_route.entry.u.msi.address_lo, 0x13);
                assert_eq!(kvm_route.entry.u.msi.data, 0x12 * i);
            }
            // However, only `update` (first vector) will actually enable be enabled
            assert_eq!(
                msix_group.vectors[i as usize]
                    .enabled
                    .load(Ordering::Acquire),
                i == 0,
            );
        }
    }

    #[test]
    fn test_msi_vector_group_persistence() {
        let vm = setup_vm_with_memory(mib_to_bytes(128));
        enable_irqchip(&vm);
        let vm = Arc::new(vm);
        let msix_group = KvmVm::create_msix_group(vm.clone(), 4).unwrap();

        enable_all_vectors(&msix_group);
        let state = msix_group.save();

        // On the real restore path the allocator state omits MSI GSIs before devices replay
        // their allocations. Mimic that here so `restore` can re-claim the saved GSIs.
        let allocator_state = vm.resource_allocator().save();
        *vm.resource_allocator() = ResourceAllocator::restore((), &allocator_state).unwrap();
        let restored_group = MsixVectorGroup::restore(vm.clone(), &state).unwrap();

        assert_eq!(msix_group.num_vectors(), restored_group.num_vectors());
        // Even if an MSI group is enabled, we don't save it as such. During restoration, the PCI
        // transport will make sure the correct config is set for the vectors and enable them
        // accordingly.
        for (id, vector) in msix_group.vectors.iter().enumerate() {
            let new_vector = &restored_group.vectors[id];
            assert_eq!(vector.gsi, new_vector.gsi);
            assert!(!new_vector.enabled.load(Ordering::Acquire));
        }

        // Both groups own the same GSIs in this test so dropping both will result in panic. Resolve
        // this by simply forgetting about the restored version.
        std::mem::forget(restored_group);
    }

    #[test]
    fn test_msi_vector_group_restore_rejects_out_of_range_gsi() {
        let vm = setup_vm_with_memory(mib_to_bytes(128));
        enable_irqchip(&vm);
        let vm = Arc::new(vm);

        // A snapshot with a GSI outside the managed MSI range must be rejected rather than panic
        // when the GSI is later freed on drop.
        let tampered_state = vec![arch::GSI_MSI_END + 1];
        MsixVectorGroup::restore(vm.clone(), &tampered_state).unwrap_err();
    }

    #[test]
    fn test_msi_vector_group_drop_frees_gsis() {
        let vm = setup_vm_with_memory(mib_to_bytes(128));
        enable_irqchip(&vm);
        let vm = Arc::new(vm);

        let gsis_before = vm.resource_allocator().allocate_gsi_msi(1).unwrap();
        for id in gsis_before.iter() {
            vm.resource_allocator()
                .gsi_msi_allocator
                .free_id(*id)
                .unwrap();
        }

        // Allocating, configuring and dropping a group must leave the allocator and the routing
        // table in the same state as before.
        {
            let group = KvmVm::create_msix_group(vm.clone(), 4).unwrap();
            let table_entry = MsixTableEntry {
                msg_addr_hi: 0x42,
                msg_addr_lo: 0x13,
                msg_data: 0x12,
                // Unmasked
                vector_ctl: 0x0,
            };
            let table_entries = vec![table_entry.clone(); 4];
            group
                .update_batched(&table_entries, PciSBDF::from(0xafa))
                .unwrap();
            assert_eq!(vm.common.interrupts.lock().unwrap().len(), 4);
        }

        assert!(vm.common.interrupts.lock().unwrap().is_empty());
        let gsis_after = vm.resource_allocator().allocate_gsi_msi(1).unwrap();
        assert_eq!(gsis_before, gsis_after);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_restore_state_resource_allocator() {
        use vm_allocator::AllocPolicy;

        let vm = setup_vm_with_memory(0x1000);
        vm.setup_irqchip().unwrap();

        // Allocate a GSI and some memory.
        let (gsi, range) = {
            let mut resource_allocator = vm.resource_allocator();

            let gsi = resource_allocator.allocate_gsi_msi(1).unwrap()[0];
            let range = resource_allocator
                .mmio32_memory
                .allocate(1024, 1024, AllocPolicy::FirstMatch)
                .unwrap();
            (gsi, range.start())
        };

        let state = vm.save_state().unwrap();
        let serialized_data = bitcode::serialize(&state).unwrap();

        let restored_state: VmState = bitcode::deserialize(&serialized_data).unwrap();
        vm.restore_state(&restored_state, false).unwrap();

        let mut resource_allocator = vm.resource_allocator();

        // The MSI GSI allocator is reset on restore (its allocations are replayed by the devices),
        // so the previously allocated GSI is free again and is the first one handed out.
        let gsi_new = resource_allocator.allocate_gsi_msi(1).unwrap()[0];
        assert_eq!(gsi, gsi_new);

        // Memory allocations, on the other hand, are restored from the serialized state, so the
        // range allocated before the snapshot is still reserved.
        resource_allocator
            .mmio32_memory
            .allocate(1024, 1024, AllocPolicy::ExactMatch(range))
            .unwrap_err();
        let range_new = resource_allocator
            .mmio32_memory
            .allocate(1024, 1024, AllocPolicy::FirstMatch)
            .unwrap();
        assert_eq!(range + 1024, range_new.start());
    }

    #[test]
    fn test_pagemap_entry_written() {
        // Each entry also carries a page frame number or swap offset in its low bits.
        let table = [
            ("present anonymous", PAGEMAP_PRESENT | 0x1234, true),
            (
                "present file",
                PAGEMAP_PRESENT | PAGEMAP_FILE_OR_SHARED | 0x1234,
                false,
            ),
            ("swapped", PAGEMAP_SWAPPED | 0x1234, true),
            ("absent", 0, false),
            ("absent with a stale frame number", 0x1234, false),
            (
                "present with bit 61 set",
                PAGEMAP_PRESENT | PAGEMAP_FILE_OR_SHARED,
                false,
            ),
        ];
        for (name, entry, written) in table {
            assert_eq!(pagemap_entry_written(entry), written, "{name}");
        }
    }

    /// The pages of a bitmap that are set.
    fn set_pages(bitmap: &[u64]) -> Vec<usize> {
        (0..bitmap.len() * 64)
            .filter(|page| (bitmap[page / 64] >> (page % 64)) & 1 == 1)
            .collect()
    }

    /// Maps `pages` pages of `fd` (or anonymous memory without it) privately, read and write.
    fn map_private(fd: Option<&File>, pages: usize) -> *mut u8 {
        let flags = match fd {
            Some(_) => libc::MAP_PRIVATE,
            None => libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
        };
        // SAFETY: Anonymous or file-backed mapping with a null hint and a valid descriptor, which
        // the tests unmap with `unmap`.
        let addr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                pages * host_page_size(),
                libc::PROT_READ | libc::PROT_WRITE,
                flags,
                fd.map_or(-1, |fd| fd.as_raw_fd()),
                0,
            )
        };
        assert_ne!(addr, libc::MAP_FAILED);
        // No huge page may make a write present in the pages around it.
        // SAFETY: The range was mapped just above.
        let ret = unsafe { libc::madvise(addr, pages * host_page_size(), libc::MADV_NOHUGEPAGE) };
        assert_eq!(ret, 0);
        addr.cast()
    }

    fn unmap(addr: *mut u8, pages: usize) {
        // SAFETY: `addr` came from `map_private` with the same number of pages.
        let ret = unsafe { libc::munmap(addr.cast(), pages * host_page_size()) };
        assert_eq!(ret, 0);
    }

    #[test]
    fn test_pagemap_bitmap_private_file_mapping() {
        const PAGES: usize = 8;
        let page_size = host_page_size();
        let mut file = TempFile::new().unwrap().into_file();
        file.write_all(&vec![0x5a; PAGES * page_size]).unwrap();
        let addr = map_private(Some(&file), PAGES);

        // The guest reads pages 1 and 3 and writes pages 2 and 5. A page that was only read stays
        // a page of the file.
        for page in [1, 3] {
            // SAFETY: `page` lies within the mapping.
            assert_eq!(unsafe { addr.add(page * page_size).read_volatile() }, 0x5a);
        }
        for page in [2, 5] {
            // SAFETY: `page` lies within the mapping.
            unsafe { addr.add(page * page_size).write_volatile(0x42) };
        }

        let mut pagemap = File::open(PAGEMAP_PATH).unwrap();
        let bitmap = pagemap_bitmap(&mut pagemap, addr, PAGES * page_size).unwrap();
        assert_eq!(set_pages(&bitmap), [2, 5]);
        unmap(addr, PAGES);
    }

    #[test]
    fn test_pagemap_bitmap_anonymous_mapping() {
        const PAGES: usize = 8;
        let page_size = host_page_size();
        let addr = map_private(None, PAGES);

        // Pages 1 and 6 are written. Page 4 is only read, which maps the zero page: it counts as
        // written and the merge of a diff punches it out, since it reads as zeros.
        for page in [1, 6] {
            // SAFETY: `page` lies within the mapping.
            unsafe { addr.add(page * page_size).write_volatile(0x42) };
        }
        // SAFETY: Page 4 lies within the mapping.
        assert_eq!(unsafe { addr.add(4 * page_size).read_volatile() }, 0);

        let mut pagemap = File::open(PAGEMAP_PATH).unwrap();
        let bitmap = pagemap_bitmap(&mut pagemap, addr, PAGES * page_size).unwrap();
        assert_eq!(set_pages(&bitmap), [1, 4, 6]);
        unmap(addr, PAGES);
    }

    #[test]
    fn test_pagemap_bitmap_spans_chunks() {
        // Two chunks of entries and a partial third, with pages written on both sides of each
        // chunk's edge.
        let pages = PAGEMAP_CHUNK_ENTRIES * 2 + 100;
        let page_size = host_page_size();
        let addr = map_private(None, pages);
        let written = [
            0,
            PAGEMAP_CHUNK_ENTRIES - 1,
            PAGEMAP_CHUNK_ENTRIES,
            2 * PAGEMAP_CHUNK_ENTRIES - 1,
            2 * PAGEMAP_CHUNK_ENTRIES,
            pages - 1,
        ];
        for page in written {
            // SAFETY: `page` lies within the mapping.
            unsafe { addr.add(page * page_size).write_volatile(0x42) };
        }

        let mut pagemap = File::open(PAGEMAP_PATH).unwrap();
        let bitmap = pagemap_bitmap(&mut pagemap, addr, pages * page_size).unwrap();
        assert_eq!(set_pages(&bitmap), written);
        unmap(addr, pages);
    }

    /// A guest whose vCPU writes pages 3, 5, 6 and 12 of its memory and reads pages 8 and 9,
    /// with dirty tracking on over a private mapping of a file. Every page that KVM's dirty log
    /// holds is in the page set that `pagemap_bitmap` reads, and both are the written pages.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_pagemap_bitmap_holds_kvm_dirty_log() {
        use kvm_ioctls::VcpuExit;

        use crate::vstate::memory::snapshot_file;

        const PAGES: usize = 16;
        const WRITTEN: [usize; 4] = [3, 5, 6, 12];
        const READ: [usize; 2] = [8, 9];
        let page_size = host_page_size();

        // A real mode program: `mov byte [addr], 0x42` for each written page, `mov al, [addr]`
        // for each page it reads, then `hlt`, which exits to this thread as no irqchip exists.
        let address = |page: usize| u16::try_from(page * page_size).unwrap().to_le_bytes();
        let mut program = Vec::new();
        for page in WRITTEN {
            program.extend([0xc6, 0x06]);
            program.extend(address(page));
            program.push(0x42);
        }
        for page in READ {
            program.push(0xa0);
            program.extend(address(page));
        }
        program.push(0xf4);

        let mut memory = vec![0x5a; PAGES * page_size];
        memory[..program.len()].copy_from_slice(&program);
        let mut file = TempFile::new().unwrap().into_file();
        file.write_all(&memory).unwrap();
        let regions = snapshot_file(
            file,
            [(GuestAddress(0), PAGES * page_size)].into_iter(),
            true,
            HugePageConfig::None,
        )
        .unwrap();
        let mut vm = setup_vm();
        vm.register_dram_memory_regions(regions).unwrap();

        let mut vcpu = vm.fd().create_vcpu(0).unwrap();
        let mut sregs = vcpu.get_sregs().unwrap();
        sregs.cs.base = 0;
        sregs.cs.selector = 0;
        vcpu.set_sregs(&sregs).unwrap();
        let mut regs = vcpu.get_regs().unwrap();
        regs.rip = 0;
        regs.rflags = 2;
        vcpu.set_regs(&regs).unwrap();
        assert!(matches!(vcpu.run().unwrap(), VcpuExit::Hlt));

        let slot = vm
            .guest_memory()
            .iter()
            .flat_map(|region| region.plugged_slots())
            .next()
            .unwrap();
        let mut pagemap = File::open(PAGEMAP_PATH).unwrap();
        let from_pagemap = pagemap_bitmap(
            &mut pagemap,
            slot.slice.ptr_guard_mut().as_ptr(),
            slot.slice.len(),
        )
        .unwrap();
        let from_kvm = vm.get_dirty_bitmap().unwrap().remove(&slot.slot).unwrap();

        let logged = set_pages(&from_kvm);
        assert_eq!(logged, WRITTEN);
        for page in &logged {
            assert!(set_pages(&from_pagemap).contains(page), "page {page}");
        }
        assert_eq!(set_pages(&from_pagemap), WRITTEN);
    }
}
