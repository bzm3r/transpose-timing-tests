#[cfg(feature = "dx12")]
extern crate gfx_backend_dx12 as dx12_back;
#[cfg(feature = "vk")]
extern crate gfx_backend_vulkan as vk_back;

extern crate gfx_hal as hal;

use crate::bitmats::BitMatrix;
use hal::{adapter::MemoryType, buffer, command, memory, pool, prelude::*, pso, query};
#[cfg(debug_assertions)]
use std::fs;
use std::{fmt, ptr, slice};

#[derive(Clone)]
pub enum KernelType {
    Threadgroup,
    Ballot,
    Shuffle,
}

impl fmt::Display for KernelType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KernelType::Threadgroup => write!(f, "{}", "threadgroup"),
            KernelType::Ballot => write!(f, "{}", "ballot"),
            KernelType::Shuffle => write!(f, "{}", "shuffle"),
        }
    }
}

#[derive(Clone)]
pub enum BackendVariant {
    Vk,
    Dx12,
}

impl fmt::Display for BackendVariant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendVariant::Vk => write!(f, "{}", "vk"),
            BackendVariant::Dx12 => write!(f, "{}", "dx12"),
        }
    }
}

#[derive(Clone)]
pub struct Task {
    pub name: String,
    pub num_bms: usize,
    pub workgroup_size: [usize; 3],
    pub kernel_type: KernelType,
    pub backend: BackendVariant,
    pub dispatch_times: Vec<f64>,
}

impl Task {
    pub fn avg_dispatch_time(&self) -> f64 {
        self.dispatch_times.iter().sum::<f64>() / (self.dispatch_times.len() as f64)
    }
}

impl fmt::Display for Task {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "name: {}\nnum_bms: {}\n workgroup size: ({}, {}, {})\n kernel type: {}, backend: {}, avg dispatch time: {} ms", self.name, self.num_bms, self.workgroup_size[0], self.workgroup_size[1], self.workgroup_size[1], self.kernel_type, self.backend, self.avg_dispatch_time())
    }
}

fn materialize_kernel(task: &Task) -> String {
    let tp = format!("kernels/transpose-{}-template.comp", task.kernel_type);
    let mut kernel =
        std::fs::read_to_string(&tp).expect(&format!("could not kernel template at path: {}", &tp));

    match task.kernel_type {
        KernelType::Threadgroup => {
            kernel = kernel.replace("~NUM_BMS~", &format!("{}", task.num_bms));
            kernel = kernel.replace("~WG_SIZE~", &format!("{}", task.workgroup_size[0]));
        }
        _ => unimplemented!(),
    }

    #[cfg(debug_assertions)]
    std::fs::write(
        format!(
            "kernels/transpose-{}-NBM={}_WGS=({},{},{}).comp",
            task.kernel_type,
            task.num_bms,
            task.workgroup_size[0],
            task.workgroup_size[1],
            task.workgroup_size[2]
        ),
        &kernel,
    )
    .unwrap();

    kernel
}

pub fn run_timing_tests(task: &mut Task, num_execs: usize) {
    match task.backend {
        BackendVariant::Vk => {
            #[cfg(not(feature = "vk"))]
            panic!("vulkan backend is not enabled");

            let instance_name = format!("vk-{}", task.kernel_type);
            #[cfg(feature = "vk")]
            execute_task::<vk_back::Backend>(instance_name, task, num_execs);
        }
        BackendVariant::Dx12 => match task.kernel_type {
            KernelType::Threadgroup => {
                #[cfg(not(feature = "dx12"))]
                panic!("dx12 backend is not loaded");

                let instance_name = format!("dx12-{}", task.kernel_type);
                #[cfg(feature = "dx12")]
                execute_task::<dx12_back::Backend>(instance_name, task, num_execs);
            }
            _ => {
                panic!("DX12 backend can only run threadgroup kernel");
            }
        },
    }
}

fn execute_task<B: hal::Backend>(instance_name: String, task: &mut Task, num_execs: usize) {
    #[cfg(debug_assertions)]
    env_logger::init();

    let bms: Vec<BitMatrix> = (0..task.num_bms).map(|_| BitMatrix::new_random()).collect();
    let tbms: Vec<BitMatrix> = bms.iter().map(|bm| bm.transpose()).collect();
    let raw_bms: Vec<[u32; 32]> = bms.iter().map(|bm| bm.as_u32s()).collect();
    let mut flat_raw_bms: Vec<u32> = Vec::new();
    for raw_bm in raw_bms.iter() {
        flat_raw_bms.extend_from_slice(&raw_bm[..]);
    }

    let instance = B::Instance::create(&instance_name, 1)
        .expect(&format!("Failed to create {} instance!", &instance_name));

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
    // we pray that this adapter's graphics/compute queues also support timestamp queries
    let family = adapter
        .queue_families
        .iter()
        .find(|family| family.queue_type().supports_compute())
        .unwrap();
    let mut gpu = unsafe {
        adapter
            .physical_device
            .open(&[(family, &[1.0])], hal::Features::empty())
            .unwrap()
    };
    let device = &gpu.device;
    let queue_group = gpu.queue_groups.first_mut().unwrap();

    let glsl = materialize_kernel(task);
    let file = glsl_to_spirv::compile(&glsl, glsl_to_spirv::ShaderType::Compute).unwrap();
    let spirv: Vec<u32> = pso::read_spirv(file).unwrap();
    let kmod = unsafe { device.create_shader_module(&spirv) }.unwrap();

    let (pipeline_layout, pipeline, set_layout, mut desc_pool) = {
        let set_layout = unsafe {
            device.create_descriptor_set_layout(
                &[pso::DescriptorSetLayoutBinding {
                    binding: 0,
                    ty: pso::DescriptorType::StorageBuffer,
                    count: 1,
                    stage_flags: pso::ShaderStageFlags::COMPUTE,
                    immutable_samplers: false,
                }],
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
                &[pso::DescriptorRangeDesc {
                    ty: pso::DescriptorType::StorageBuffer,
                    count: 1,
                }],
                pso::DescriptorPoolCreateFlags::empty(),
            )
        }
        .expect("Can't create descriptor pool");
        (pipeline_layout, pipeline, set_layout, desc_pool)
    };

    let stride = std::mem::size_of::<u32>() as u64;
    let (staging_mem, staging_buf, staging_size) = unsafe {
        create_buffer::<B>(
            &device,
            &memory_properties.memory_types,
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
    }

    let (device_mem, device_buf, _device_buffer_size) = unsafe {
        create_buffer::<B>(
            &device,
            &memory_properties.memory_types,
            memory::Properties::DEVICE_LOCAL,
            buffer::Usage::TRANSFER_SRC | buffer::Usage::TRANSFER_DST | buffer::Usage::STORAGE,
            stride,
            flat_raw_bms.len() as u64,
        )
    };

    let mut cmd_pool =
        unsafe { device.create_command_pool(family.id(), pool::CommandPoolCreateFlags::empty()) }
            .expect("Can't create command pool");
    let query_pool = unsafe { device.create_query_pool(query::Type::Timestamp, 2).unwrap() };
    let fence = device.create_fence(false).unwrap();

    assert_eq!(task.num_bms % task.workgroup_size[0], 0);
    let num_dispatch_groups = task.num_bms / task.workgroup_size[0];
    for _ in 0..num_execs {
        unsafe {
            let desc_set = desc_pool.allocate_set(&set_layout).unwrap();
            device.write_descriptor_sets(Some(pso::DescriptorSetWrite {
                set: &desc_set,
                binding: 0,
                array_offset: 0,
                descriptors: Some(pso::Descriptor::Buffer(&device_buf, None..None)),
            }));
            let mut cmd_buf = cmd_pool.allocate_one(command::Level::Primary);
            cmd_buf.begin_primary(command::CommandBufferFlags::ONE_TIME_SUBMIT);
            cmd_buf.reset_query_pool(&query_pool, 0..2);
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
            cmd_buf.bind_compute_descriptor_sets(&pipeline_layout, 0, &[desc_set], &[]);
            cmd_buf.write_timestamp(
                pso::PipelineStage::COMPUTE_SHADER,
                query::Query {
                    pool: &query_pool,
                    id: 0,
                },
            );
            cmd_buf.dispatch([num_dispatch_groups as u32, 1, 1]);
            cmd_buf.write_timestamp(
                pso::PipelineStage::COMPUTE_SHADER,
                query::Query {
                    pool: &query_pool,
                    id: 1,
                },
            );
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
                    size: stride * bms.len() as u64,
                }],
            );
            cmd_buf.finish();

            queue_group.queues[0].submit_without_semaphores(Some(&cmd_buf), Some(&fence));

            device.wait_for_fence(&fence, !0).unwrap();
            cmd_pool.free(Some(cmd_buf));
        }

        let (result, ts) = unsafe {
            let mapping = device.map_memory(&staging_mem, 0..staging_size).unwrap();
            let r = Vec::<u32>::from(slice::from_raw_parts::<u32>(
                mapping as *const u8 as *const u32,
                32 * bms.len(),
            ));
            device.unmap_memory(&staging_mem);

            let mut t = vec![0u32; 2];
            // what is the purpose of this wait?
            device.wait_idle().unwrap();
            let raw_t = slice::from_raw_parts_mut(t.as_mut_ptr() as *mut u8, 4 * 2);
            device
                .get_query_pool_results(&query_pool, 0..2, raw_t, 4, query::ResultFlags::WAIT)
                .unwrap();

            (r, t)
        };

        assert_eq!(flat_raw_bms.len(), result.len());
        let mut result_bms: Vec<BitMatrix> = (0..task.num_bms)
            .map(|i| {
                BitMatrix::from_u32s(&result[i * 32..(i + 1) * 32])
                    .expect("could not construct BitMatrix from u32 slice")
            })
            .collect();
        println!("{}", &bms[0].transpose());
        println!("{}", &result_bms[0]);
        assert!(bms
            .iter()
            .zip(result_bms.iter())
            .all(|(bm, rbm)| bm.transpose().identical_to(rbm)));
        println!("GPU results verified!");
        task.dispatch_times.push((ts[1] - ts[0]) as f64);
    }

    unsafe {
        device.destroy_command_pool(cmd_pool);
        device.destroy_descriptor_pool(desc_pool);
        device.destroy_descriptor_set_layout(set_layout);
        device.destroy_shader_module(kmod);
        device.destroy_buffer(device_buf);
        device.destroy_buffer(staging_buf);
        device.destroy_fence(fence);
        device.destroy_pipeline_layout(pipeline_layout);
        device.free_memory(device_mem);
        device.free_memory(staging_mem);
        device.destroy_compute_pipeline(pipeline);
    }
}

unsafe fn create_buffer<B: hal::Backend>(
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
            requirements.type_mask & (1 << id) != 0 && memory_type.properties.contains(properties)
        })
        .unwrap()
        .into();

    let memory = device.allocate_memory(ty, requirements.size).unwrap();
    device.bind_buffer_memory(&memory, 0, &mut buffer).unwrap();

    (memory, buffer, requirements.size)
}
