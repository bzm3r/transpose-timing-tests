#[cfg(feature = "vk")]
extern crate gfx_backend_vulkan as Vulkan;

#[cfg(feature = "dx12")]
extern crate gfx_backend_dx12 as Dx12;

#[cfg(feature = "metal")]
extern crate gfx_backend_metal as Metal;

extern crate csv;
extern crate gfx_hal as hal;
extern crate shaderc;

use crate::bitmats::{BitMatrix, Interpretation};
use crate::task::{KernelType, NumCpuExecs, NumGpuExecs, Task, TaskGroup, TaskGroupDefn};
use hal::{
    adapter::{Adapter, MemoryProperties, MemoryType},
    buffer, command, memory, pool,
    prelude::*,
    pso, query, Instance,
};
use std::{ptr, slice};

use self::hal::queue::QueueFamily;
use std::fmt;
use std::fmt::Formatter;
use std::fs::OpenOptions;

#[derive(Clone)]
pub enum BackendVariant {
    #[cfg(feature = "vk")]
    Vk,
    #[cfg(feature = "dx12")]
    Dx12,
    #[cfg(feature = "metal")]
    Metal,
}

impl fmt::Display for BackendVariant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            #[cfg(feature = "vk")]
            BackendVariant::Vk => write!(f, "{}", "vk"),
            #[cfg(feature = "dx12")]
            BackendVariant::Dx12 => write!(f, "{}", "dx12"),
            #[cfg(feature = "metal")]
            BackendVariant::Metal => write!(f, "{}", "metal"),
            _ => write!(f, "{}", "unknown"),
        }
    }
}

const NVIDIA_GTX_1060: &str = "GeForce GTX 1060";
const NVIDIA_RTX_2060: &str = "GeForce RTX 2060";
const INTEL_HD_630: &str = "Intel(R) HD Graphics 630";
const INTEL_IVYBRIDGE_MOBILE: &str = "Intel(R) Ivybridge Mobile";
const INTEL_IRIS_PLUS_640: &str = "Intel(R) Iris(TM) Plus Graphics 640";
const INTEL_HD_520: &str = "Intel(R) HD Graphics 520";
const AMD_RADEON_RX570: &str = "Radeon RX 570 Series";

pub type DeviceName = String;

pub struct GpuTestEnv<B: hal::Backend> {
    pub backend: BackendVariant,
    pub instance: B::Instance,
    pub device_name: DeviceName,
    adapter: Adapter<B>,
    memory_properties: MemoryProperties,
    ts_grain: f64,
    task_group: Option<TaskGroup>,
}

impl<B: hal::Backend> GpuTestEnv<B> {
    fn load(instance: &B::Instance) -> (DeviceName, MemoryProperties, Adapter<B>) {
        let adapter = instance
            .enumerate_adapters()
            .into_iter()
            .find(|a| {
                a.queue_families
                    .iter()
                    .any(|family| family.queue_type().supports_compute())
            })
            .expect("Failed to find a GPU with compute support!");

        let memory_properties = adapter.physical_device.memory_properties();
        let device_name = adapter.info.name.clone();

        (device_name, memory_properties, adapter)
    }

    unsafe fn create_buffer(
        device: &B::Device,
        memory_types: &[MemoryType],
        properties: memory::Properties,
        usage: buffer::Usage,
        stride: u64,
        len: u64,
    ) -> (B::Memory, B::Buffer, u64) {
        let mut buffer = device.create_buffer(stride * len, usage).unwrap();
        let requirements = device.get_buffer_requirements(&buffer);

        let ty = memory_types
            .into_iter()
            .enumerate()
            .position(|(id, memory_type)| {
                requirements.type_mask & (1 << id) != 0
                    && memory_type.properties.contains(properties)
            })
            .unwrap()
            .into();

        let memory = device.allocate_memory(ty, requirements.size).unwrap();
        device.bind_buffer_memory(&memory, 0, &mut buffer).unwrap();

        (memory, buffer, requirements.size)
    }

    pub fn time_task(
        &self,
        num_cpu_execs: NumCpuExecs,
        num_gpu_execs: NumGpuExecs,
        task: &mut Task,
    ) {
        // we pray that this adapter's graphics/compute queues also support timestamp queries
        let family = self
            .adapter
            .queue_families
            .iter()
            .find(|family| family.queue_type().supports_compute())
            .unwrap();
        let mut gpu = unsafe {
            self.adapter
                .physical_device
                .open(&[(family, &[1.0])], hal::Features::empty())
                .unwrap()
        };
        let device = &gpu.device;
        let queue_group = gpu.queue_groups.first_mut().unwrap();

        let interpretation = match task.kernel_type {
            KernelType::Shuffle8 | KernelType::Threadgroup1d8 | KernelType::Threadgroup2d8 => {
                Interpretation::B8
            },
            _ => {
                Interpretation::B32
            }
        };

        let mut bms: Vec<BitMatrix> = (0..task.num_bms).map(|i| BitMatrix::new_random(interpretation)).collect();
        let raw_bms: Vec<[u32; 32]> = bms.iter().map(|bm| bm.as_u32s()).collect();
        let mut flat_raw_bms: Vec<u32> = Vec::new();
        for raw_bm in raw_bms.iter() {
            flat_raw_bms.extend_from_slice(&raw_bm[..]);
        }

        let kmod = {
            let compiled_kernel = task.compile_kernel();
            let spirv: Vec<u32> = pso::read_spirv(&compiled_kernel).unwrap();
            unsafe { device.create_shader_module(&spirv) }.unwrap()
        };

        let (pipeline_layout, pipeline, set_layout, mut desc_pool) = {
            let set_layout = unsafe {
                device.create_descriptor_set_layout(
                    &[
                        pso::DescriptorSetLayoutBinding {
                            binding: 0,
                            ty: pso::DescriptorType::Buffer {
                                ty: pso::BufferDescriptorType::Storage { read_only: false },
                                format: pso::BufferDescriptorFormat::Structured {
                                    dynamic_offset: false,
                                },
                            },
                            count: 1,
                            stage_flags: pso::ShaderStageFlags::COMPUTE,
                            immutable_samplers: false,
                        },
                        pso::DescriptorSetLayoutBinding {
                            binding: 1,
                            ty: pso::DescriptorType::Buffer {
                                ty: pso::BufferDescriptorType::Uniform,
                                format: pso::BufferDescriptorFormat::Structured {
                                    dynamic_offset: false,
                                },
                            },
                            count: 1,
                            stage_flags: pso::ShaderStageFlags::COMPUTE,
                            immutable_samplers: false,
                        },
                    ],
                    &[],
                )
            }
            .expect("Can't create descriptor set layout");

            let pipeline_layout = unsafe { device.create_pipeline_layout(Some(&set_layout), &[]) }
                .expect("Can't create pipeline layout");
            let entry_point = pso::EntryPoint {
                entry: "main",
                module: &kmod,
                specialization: pso::Specialization::default(),
            };
            let pipeline = unsafe {
                device.create_compute_pipeline(
                    &pso::ComputePipelineDesc::new(entry_point, &pipeline_layout),
                    None,
                )
            }
            .expect("Error creating compute pipeline!");

            let desc_pool = unsafe {
                device.create_descriptor_pool(
                    1,
                    &[
                        pso::DescriptorRangeDesc {
                            ty: pso::DescriptorType::Buffer {
                                ty: pso::BufferDescriptorType::Storage { read_only: false },
                                format: pso::BufferDescriptorFormat::Structured {
                                    dynamic_offset: false,
                                },
                            },
                            count: 1,
                        },
                        pso::DescriptorRangeDesc {
                            ty: pso::DescriptorType::Buffer {
                                ty: pso::BufferDescriptorType::Uniform,
                                format: pso::BufferDescriptorFormat::Structured {
                                    dynamic_offset: false,
                                },
                            },
                            count: 1,
                        },
                    ],
                    pso::DescriptorPoolCreateFlags::empty(),
                )
            }
            .expect("Can't create descriptor pool");
            (pipeline_layout, pipeline, set_layout, desc_pool)
        };

        let stride = std::mem::size_of::<u32>() as u64;
        let (uniform_mem, uniform_buf, uniform_size) = unsafe {
            Self::create_buffer(
                device,
                &self.memory_properties.memory_types,
                memory::Properties::CPU_VISIBLE,
                buffer::Usage::UNIFORM,
                stride,
                2,
            )
        };

        let (staging_mem, staging_buf, staging_size) = unsafe {
            Self::create_buffer(
                device,
                &self.memory_properties.memory_types,
                memory::Properties::CPU_VISIBLE | memory::Properties::COHERENT,
                buffer::Usage::TRANSFER_SRC | buffer::Usage::TRANSFER_DST,
                stride,
                flat_raw_bms.len() as u64,
            )
        };

        unsafe {
            let mapping = device.map_memory(&staging_mem, 0..staging_size).unwrap();
            ptr::copy_nonoverlapping(
                flat_raw_bms.as_ptr() as *const u8,
                mapping,
                flat_raw_bms.len() * stride as usize,
            );
            device.unmap_memory(&staging_mem);

            let mapping = device.map_memory(&uniform_mem, 0..uniform_size).unwrap();
            ptr::copy_nonoverlapping(
                vec![task.num_bms, num_gpu_execs.0].as_ptr() as *const u8,
                mapping,
                2 * stride as usize,
            );
            device.unmap_memory(&uniform_mem);
        }

        let (device_mem, device_buf, _device_buffer_size) = unsafe {
            Self::create_buffer(
                device,
                &self.memory_properties.memory_types,
                memory::Properties::DEVICE_LOCAL,
                buffer::Usage::TRANSFER_SRC | buffer::Usage::TRANSFER_DST | buffer::Usage::STORAGE,
                stride,
                flat_raw_bms.len() as u64,
            )
        };

        let desc_set = unsafe {
            let ds = desc_pool.allocate_set(&set_layout).unwrap();
            device.write_descriptor_sets(Some(pso::DescriptorSetWrite {
                set: &ds,
                binding: 0,
                array_offset: 0,
                descriptors: Some(pso::Descriptor::Buffer(&device_buf, None..None)),
            }));
            device.write_descriptor_sets(Some(pso::DescriptorSetWrite {
                set: &ds,
                binding: 1,
                array_offset: 0,
                descriptors: Some(pso::Descriptor::Buffer(&uniform_buf, None..None)),
            }));
            ds
        };

        let mut cmd_pool = unsafe {
            device.create_command_pool(family.id(), pool::CommandPoolCreateFlags::empty())
        }
        .expect("Can't create command pool");
        let query_pool = unsafe { device.create_query_pool(query::Type::Timestamp, 2).ok() };
        let fence = device.create_fence(false).unwrap();

        let num_dispatch_groups = {
            let num_mats_per_wg = task.workgroup_size[0]*task.workgroup_size[1] / 32;
            (task.num_bms + num_mats_per_wg - 1) / num_mats_per_wg
        };
        println!(
            "num bms: {}, num dispatch groups: {}",
            task.num_bms, num_dispatch_groups
        );
        for i in 0..num_cpu_execs.0 {
            unsafe {
                let mut cmd_buf = cmd_pool.allocate_one(command::Level::Primary);
                cmd_buf.begin_primary(command::CommandBufferFlags::ONE_TIME_SUBMIT);
                if let Some(query_pool) = query_pool.as_ref() {
                    cmd_buf.reset_query_pool(&query_pool, 0..2);
                }
                cmd_buf.copy_buffer(
                    &staging_buf,
                    &device_buf,
                    &[command::BufferCopy {
                        src: 0,
                        dst: 0,
                        size: stride * flat_raw_bms.len() as u64,
                    }],
                );
                cmd_buf.pipeline_barrier(
                    pso::PipelineStage::TRANSFER..pso::PipelineStage::COMPUTE_SHADER,
                    memory::Dependencies::empty(),
                    Some(memory::Barrier::Buffer {
                        states: buffer::Access::TRANSFER_WRITE
                            ..buffer::Access::SHADER_READ | buffer::Access::SHADER_WRITE,
                        families: None,
                        target: &device_buf,
                        range: None..None,
                    }),
                );
                cmd_buf.bind_compute_pipeline(&pipeline);
                cmd_buf.bind_compute_descriptor_sets(
                    &pipeline_layout,
                    0,
                    [&desc_set].iter().cloned(),
                    &[],
                );
                if let Some(query_pool) = query_pool.as_ref() {
                    cmd_buf.write_timestamp(
                        pso::PipelineStage::COMPUTE_SHADER,
                        query::Query {
                            pool: &query_pool,
                            id: 0,
                        },
                    );
                }
                cmd_buf.dispatch([num_dispatch_groups as u32, 1, 1]);
                if let Some(query_pool) = query_pool.as_ref() {
                    cmd_buf.write_timestamp(
                        pso::PipelineStage::COMPUTE_SHADER,
                        query::Query {
                            pool: &query_pool,
                            id: 1,
                        },
                    );
                }
                cmd_buf.pipeline_barrier(
                    pso::PipelineStage::COMPUTE_SHADER..pso::PipelineStage::TRANSFER,
                    memory::Dependencies::empty(),
                    Some(memory::Barrier::Buffer {
                        states: buffer::Access::SHADER_READ | buffer::Access::SHADER_WRITE
                            ..buffer::Access::TRANSFER_READ,
                        families: None,
                        target: &device_buf,
                        range: None..None,
                    }),
                );
                cmd_buf.copy_buffer(
                    &device_buf,
                    &staging_buf,
                    &[command::BufferCopy {
                        src: 0,
                        dst: 0,
                        size: stride * flat_raw_bms.len() as u64,
                    }],
                );
                cmd_buf.finish();

                let start = std::time::Instant::now();
                queue_group.queues[0].submit_without_semaphores(Some(&cmd_buf), Some(&fence));

                device.wait_for_fence(&fence, !0).unwrap();
                device.reset_fence(&fence).unwrap();
                task.instant_times
                    .push(start.elapsed().as_micros() as f64 * 1e-3);
                cmd_pool.free(Some(cmd_buf));
            }

            if i == 0 {
                let result = unsafe {
                    let mapping = device.map_memory(&staging_mem, 0..staging_size).unwrap();
                    let r = Vec::<u32>::from(slice::from_raw_parts::<u32>(
                        mapping as *const u8 as *const u32,
                        flat_raw_bms.len(),
                    ));
                    device.unmap_memory(&staging_mem);
                    r
                };

                assert_eq!(flat_raw_bms.len(), result.len());
                let result_bms: Vec<BitMatrix> = (0..(task.num_bms as usize))
                    .map(|i| {
                        BitMatrix::from_u32s(&result[i * 32..(i + 1) * 32], interpretation)
                            .expect("could not construct BitMatrix from u32 slice")
                    })
                    .collect();

                // for i in 0..2 {
                //     println!("input bm {}: {}", i, &bms[i]);
                //     println!("rbm {}: {}", i, &result_bms[i]);
                //     //println!("expected {}: {}", i, bms[i].transpose());
                // }

                for (i, (bm, rbm)) in bms.iter().zip(result_bms.iter()).enumerate() {
                    if !(bm.transpose().identical_to(rbm)) {
                        task.delete_compiled_kernel();
                        panic!(
                            "GPU result {} incorrect!\ninput: {}\nexpected:{}\ngot: {}",
                            i,
                            &bms[i],
                            &bms[i].transpose(),
                            &result_bms[i]
                        );
                    }
                }
                bms = result_bms;
                println!("GPU results verified!");
            }

            let ts = unsafe {
                let mut ts = vec![0u32; 2];
                if let Some(query_pool) = query_pool.as_ref() {
                    let raw_t = slice::from_raw_parts_mut(ts.as_mut_ptr() as *mut u8, 4 * 2);
                    device
                        .get_query_pool_results(
                            &query_pool,
                            0..2,
                            raw_t,
                            4,
                            query::ResultFlags::WAIT,
                        )
                        .unwrap();
                }
                ts
            };
            task.timestamp_query_times
                .push((ts[1].wrapping_sub(ts[0])) as f64 * self.ts_grain);
        }

        unsafe {
            if let Some(query_pool) = query_pool {
                device.destroy_query_pool(query_pool);
            }
            device.destroy_command_pool(cmd_pool);
            device.destroy_descriptor_pool(desc_pool);
            device.destroy_descriptor_set_layout(set_layout);
            device.destroy_shader_module(kmod);
            device.destroy_buffer(device_buf);
            device.destroy_buffer(staging_buf);
            device.destroy_buffer(uniform_buf);
            device.destroy_fence(fence);
            device.destroy_pipeline_layout(pipeline_layout);
            device.free_memory(device_mem);
            device.free_memory(staging_mem);
            device.free_memory(uniform_mem);
            device.destroy_compute_pipeline(pipeline);
        }
    }

    #[cfg(feature = "vk")]
    pub fn vulkan() -> GpuTestEnv<Vulkan::Backend> {
        let instance = Vulkan::Instance::create("vk-back", 1, hal::ApiVersion::new(1, 1, 1))
            .expect(&format!("could not create Vulkan instance"));

        let (device_name, memory_properties, adapter) = GpuTestEnv::load(&instance);
        let ts_grain = vk_get_timestamp_period(&device_name).unwrap();
        GpuTestEnv {
            backend: BackendVariant::Vk,
            instance,
            device_name,
            adapter,
            memory_properties,
            ts_grain,
            task_group: None,
        }
    }

    #[cfg(feature = "dx12")]
    pub fn dx12() -> GpuTestEnv<Dx12::Backend> {
        let instance = Dx12::Instance::create("dx12-back", 1, hal::ApiVersion::dummy())
            .expect(&format!("could not create DX12 instance"));

        let (device_name, memory_properties, adapter) = GpuTestEnv::load(&instance);
        let ts_grain = dx12_get_timestamp_period(&device_name).unwrap();
        GpuTestEnv {
            backend: BackendVariant::Vk,
            instance,
            device_name,
            adapter,
            memory_properties,
            ts_grain,
            task_group: None,
        }
    }

    #[cfg(feature = "metal")]
    pub fn metal() -> GpuTestEnv<Metal::Backend> {
        let instance = Metal::Instance::create("metal-back", 1, hal::ApiVersion::dummy())
            .expect(&format!("could not create Metal instance"));

        let (device_name, memory_properties, adapter) = GpuTestEnv::load(&instance);
        let ts_grain = 1.0;
        GpuTestEnv {
            backend: BackendVariant::Metal,
            instance,
            device_name,
            adapter,
            memory_properties,
            ts_grain,
            task_group: None,
        }
    }

    pub fn set_task_group(&mut self, task_group_defn: TaskGroupDefn) {
        self.task_group = match task_group_defn.kernel_type {
            KernelType::Threadgroup2d32 | KernelType::Threadgroup2d8 => {
                let task_group_prefix = format!("{}-{}", self.backend, task_group_defn.kernel_type);
                Some(TaskGroup {
                    name: format!("{}-{}", &task_group_prefix, self.device_name),
                    num_gpu_execs: task_group_defn.num_gpu_execs,
                    num_cpu_execs: task_group_defn.num_cpu_execs,
                    kernel_type: task_group_defn.kernel_type,
                    tasks: {
                        let mut tasks = Vec::<Task>::new();

                        for n in 0u32..6 {
                            let num_wg = 2u32.pow(n);
                            for num_bms in (0u32..15).map(|u| 2u32.pow(u)) {
                                tasks.push(Task {
                                    name: format!(
                                        "{}-NBMS={}-WGS=({}, 32)",
                                        &task_group_prefix,
                                        num_bms,
                                        num_wg,
                                    ),
                                    num_bms,
                                    workgroup_size: [num_wg, 32],
                                    timestamp_query_times: vec![],
                                    instant_times: vec![],
                                    kernel_name: format!(
                                        "transpose-{}-WGS=({},{})",
                                        task_group_defn.kernel_type,
                                        num_wg,
                                        32
                                    ),
                                    kernel_type: task_group_defn.kernel_type,
                                })
                            }
                        }

                        tasks
                    },
                })
            }
            KernelType::Shuffle8 | KernelType::Shuffle32 | KernelType::Ballot32 => {
                if check_kernel_for_intel(&self.device_name, &task_group_defn.kernel_type).unwrap() {
                    println!(
                        "Detected Intel device, skipping creation of SIMD-N (N > 8) kernel task group."
                    );
                    None
                } else {
                    let task_group_prefix = format!("{}-{}", self.backend, task_group_defn.kernel_type);
                    Some(TaskGroup {
                        name: format!("{}-{}", &task_group_prefix, self.device_name),
                        num_gpu_execs: task_group_defn.num_gpu_execs,
                        num_cpu_execs: task_group_defn.num_cpu_execs,
                        kernel_type: task_group_defn.kernel_type,
                        tasks: {
                            let mut tasks = Vec::<Task>::new();

                            for n in 5..11 {
                                let num_threads = 2u32.pow(n);
                                for num_bms in (0u32..15).map(|u| 2u32.pow(u)) {
                                    tasks.push(Task {
                                        name: format!(
                                            "{}-NBMS={}-WGS=({},{})",
                                            &task_group_prefix, num_bms, num_threads, 1
                                        ),
                                        num_bms,
                                        workgroup_size: [num_threads, 1],
                                        timestamp_query_times: vec![],
                                        instant_times: vec![],
                                        kernel_name: format!(
                                            "transpose-{}-WGS=({},{})",
                                            task_group_defn.kernel_type, num_threads, 1
                                        ),
                                        kernel_type: task_group_defn.kernel_type,
                                    })
                                }
                            }
                            tasks
                        },
                    })
                }
            }
            KernelType::Threadgroup1d32 | KernelType::Threadgroup1d8 | KernelType::HybridShuffle32 | KernelType::HybridShuffleAdaptive32 => {
                let task_group_prefix = format!("{}-{}", self.backend, task_group_defn.kernel_type);
                Some(TaskGroup {
                    name: format!("{}-{}", &task_group_prefix, self.device_name),
                    num_gpu_execs: task_group_defn.num_gpu_execs,
                    num_cpu_execs: task_group_defn.num_cpu_execs,
                    kernel_type: task_group_defn.kernel_type,
                    tasks: {
                        let mut tasks = Vec::<Task>::new();

                        for n in 5u32..11 {
                            let num_threads = 2u32.pow(n);
                            // can't go from 0 to 15 yet due to hybrid shuffle kernel issues (see branch barrier-weirdness)
                            for num_bms in ((n - 5)..15).map(|u| 2u32.pow(u)) {
                                tasks.push(Task {
                                    name: format!(
                                        "{}-NBMS={}-WGS=({},{})",
                                        &task_group_prefix, num_bms, num_threads, 1
                                    ),
                                    num_bms,
                                    workgroup_size: [num_threads, 1],
                                    timestamp_query_times: vec![],
                                    instant_times: vec![],
                                    kernel_name: format!(
                                        "transpose-{}-WGS=({},{})",
                                        task_group_defn.kernel_type,
                                        num_threads,
                                        1
                                    ),
                                    kernel_type: task_group_defn.kernel_type,
                                })
                            }
                        }

                        tasks
                    },
                })
            },
        }
    }

    pub fn time_task_group(&mut self) {
        println!("{}", self);

        let timed_tasks = match self.task_group.as_ref() {
            Some(tg) => {
                println!("{}", tg);
                tg.tasks
                    .iter()
                    .map(|t| {
                        let mut nt = t.clone();
                        self.time_task(tg.num_cpu_execs, tg.num_gpu_execs, &mut nt);
                        println!("{}", nt);
                        nt
                    })
                    .collect::<Vec<Task>>()
            }
            _ => {
                println!("no task group to run");
                vec![]
            }
        };

        match self.task_group.as_mut() {
            Some(tg) => {
                tg.tasks = timed_tasks;
            }
            _ => {}
        }
    }

    pub fn save_results(&self) {
        match &self.task_group {
            Some(tg) => {
                let fp = format!("plots/{}.dat", tg.name);
                match OpenOptions::new().append(true).write(true).open(&fp) {
                    Ok(f) => {
                        let mut wtr = csv::WriterBuilder::new().flexible(true).from_writer(f);
                        for task in tg.tasks.iter() {
                            wtr.write_record(to_string_vec(&[
                                task.workgroup_size[0],
                                task.workgroup_size[1],
                                task.num_bms,
                            ]))
                            .unwrap();
                            wtr.write_record(to_string_vec(
                                &task
                                    .timestamp_query_times
                                    .iter()
                                    .map(|&t| {
                                        (task.num_bms as f64 * tg.num_gpu_execs.0 as f64)
                                            / (t * 1e-3)
                                    })
                                    .collect::<Vec<f64>>(),
                            ))
                            .unwrap();
                            wtr.write_record(to_string_vec(
                                &task
                                    .instant_times
                                    .iter()
                                    .map(|&t| {
                                        (task.num_bms as f64 * tg.num_gpu_execs.0 as f64)
                                            / (t * 1e-3)
                                    })
                                    .collect::<Vec<f64>>(),
                            ))
                            .unwrap();
                        }
                        wtr.flush().unwrap();
                    }
                    _ => {
                        let f = OpenOptions::new()
                            .create(true)
                            .write(true)
                            .open(&fp)
                            .unwrap();
                        let mut wtr = csv::WriterBuilder::new().flexible(true).from_writer(f);
                        wtr.write_record(&[self.device_name.clone()]).unwrap();
                        for task in tg.tasks.iter() {
                            wtr.write_record(to_string_vec(&[
                                task.workgroup_size[0],
                                task.workgroup_size[1],
                                task.num_bms,
                            ]))
                            .unwrap();
                            wtr.write_record(to_string_vec(
                                &task
                                    .timestamp_query_times
                                    .iter()
                                    .map(|&t| {
                                        (task.num_bms as f64 * tg.num_gpu_execs.0 as f64)
                                            / (t * 1e-3)
                                    })
                                    .collect::<Vec<f64>>(),
                            ))
                            .unwrap();
                            wtr.write_record(to_string_vec(
                                &task
                                    .instant_times
                                    .iter()
                                    .map(|&t| {
                                        (task.num_bms as f64 * tg.num_gpu_execs.0 as f64)
                                            / (t * 1e-3)
                                    })
                                    .collect::<Vec<f64>>(),
                            ))
                            .unwrap();
                        }
                        wtr.flush().unwrap();
                    }
                };
            }
            _ => {
                println!("no task group with results to save");
            }
        }
    }
}

impl<B: hal::Backend> fmt::Display for GpuTestEnv<B> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "backend: {}, device: {}", self.backend, self.device_name)
    }
}

fn vk_get_timestamp_period(device_name: &str) -> Result<f64, String> {
    match device_name {
        NVIDIA_GTX_1060 => {
            // https://vulkan.gpuinfo.org/displayreport.php?id=7922
            Ok(1.0e-6)
        }
        NVIDIA_RTX_2060 => {
            // https://vulkan.gpuinfo.org/displayreport.php?id=7885
            Ok(1.0e-6)
        }
        INTEL_HD_630 => {
            // https://vulkan.gpuinfo.org/displayreport.php?id=7797
            Ok(83.333e-6)
        }
        INTEL_IVYBRIDGE_MOBILE => {
            // https://vulkan.gpuinfo.org/displayreport.php?id=7929
            Ok(80.0e-6)
        }
        INTEL_IRIS_PLUS_640 => {
            // https://vulkan.gpuinfo.org/displayreport.php?id=7855
            Ok(83.333e-6)
        }
        INTEL_HD_520 => {
            // https://vulkan.gpuinfo.org/displayreport.php?id=7751
            Ok(83.333e-6)
        }
        AMD_RADEON_RX570 => {
            // https://vulkan.gpuinfo.org/displayreport.php?id=7941
            Ok(40.0e-6)
        }
        _ => {
            let err_string = format!(
                "timestamp_period data unavailable for {}. Please update!",
                device_name
            );
            println!("{}", &err_string);
            Err(err_string)
        }
    }
}

pub fn check_kernel_for_intel(device_name: &str, kernel_type: &KernelType) -> Result<bool, String> {
    match device_name {
        NVIDIA_GTX_1060 | NVIDIA_RTX_2060 | AMD_RADEON_RX570 => Ok(false),
        INTEL_HD_520 | INTEL_HD_630 | INTEL_IRIS_PLUS_640 | INTEL_IVYBRIDGE_MOBILE => {
            match kernel_type {
                KernelType::Shuffle32 | KernelType::Ballot32 => {
                    Ok(true)
                },
                _ => {
                    Ok(false)
                }
            }
        },
        _ => Err(String::from("Unknown device.")),
    }
}

#[allow(dead_code)]
fn dx12_get_timestamp_period(_device_name: &str) -> Result<f64, String> {
    Err(String::from(
        "unable to determine timestamp period using gfx-rs for dx12 backend",
    ))
}

fn to_string_vec<T: fmt::Display>(input: &[T]) -> Vec<String> {
    input.iter().map(|i| format!("{}", i)).collect()
}
