#[cfg(feature = "dx12")]
extern crate gfx_backend_dx12 as dx12back;
#[cfg(feature = "vulkan")]
extern crate gfx_backend_vulkan as vkback;

extern crate gfx_hal as hal;

use crate::bitmats::BitMatrix;
use hal::{adapter::MemoryType, buffer, command, memory, pool, prelude::*, pso};
use std::{
    fmt, fs,
    path::{Path, PathBuf},
    ptr, slice,
    str::FromStr,
};

pub enum KernelType {
    Threadgroup,
    Ballot,
    Shuffle,
}

impl fmt::Display for KernelType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Kernel::Threadgroup(_) => write!(f, "threadgroup", self.x, self.y),
            Kernel::Ballot => write!(f, "ballot", self.x, self.y),
            Kernel::Shuffle => write!(f, "shuffle", self.x, self.y),
        }
    }
}

pub enum BackendVariant {
    Vk,
    Dx12,
}

pub struct Task {
    num_bms: usize,
    workgroup_size: [usize; 3],
    kernel_type: KernelType,
    backend: BackendVariant,
    dispatch_times: Vec<f64>,
}

fn materialize_kernel(task: &Task) -> String {
    let tp = format!("kernels/transpose-{}-template.comp", task.kernel_type);
    let mut kernel =
        std::fs::read_to_string(&tp).expect(&format!("could not kernel template at path: {}", &tp));

    match task.kernel_type {
        KernelType::Threadgroup => {
            kernel = kernel.replace("~NUM_BMS~", &format!("{}", task.num_bms));
            kernel = kernel.replace("~WG_SIZE~", &format!("{}", task.workgroup_size));
        }
        _ => unimplemented!(),
    }
    #[cfg(debug_assertions)]
    std::fs::write(
        format!(
            "kernels/transpose-{}-NBM={}_WGS={}.comp",
            task.kernel_type, task.num_bms, task.workgroup_size
        ),
        &kernel,
    );

    kernel
}

pub fn run_timing_tests(mut task: Task, num_execs: usize) -> Task {
    match &task.backend {
        BackendVariant::Vk => {
            let instance_name = format!("Vk-{}", &task.kernel_type);
            run_tests::<vk::Backend>(instance_name, &mut task, num_execs);
        }
        BackendVariant::Dx12 => match &task.kernel_type {
            KernelType::Threadgroup => {
                let instance_name = format!("Vk-{}", &task.kernel_type);
                run_tests::<vk::Backend>(instance_name, &mut task, num_execs);
            }
            _ => {
                panic!("DX12 backend can only run threadgroup kernel");
            }
        },
    }

    task
}

pub fn run_tests<B: hal::Backend>(
    instance_name: String,
    task: &mut Task,
    num_execs: usize,
) -> Vec<DispatchTime> {
    #[cfg(debug_assertions)]
    env_logger::init();

    let bms: Vec<BitMatrix> = (0..task.num_bms)
        .iter()
        .map(|_| BitMatrix::new_random())
        .collect();
    let tbms: Vec<BitMatrix> = bms.iter().map(|bm| bm.transpose()).colelct();
    let u32_bms: Vec<[u32; 32]> = bms.iter().map(|bm| bm.as_u32s()).collect();
    let mut flat_u32_bms: Vec<u32> = Vec::new();
    u32_bms
        .iter()
        .map(|bm| flat_u32_bms.extend_from_slice(&bm.as_u32s()));

    let instance = B::Instance::create(&instance_name, 1)
        .expect(format!("Failed to create {} instance!", &instance_name));

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
                    ty: pso::DescriptorType::Buffer {
                        ty: pso::BufferDescriptorType::Storage { read_only: false },
                        format: pso::BufferDescriptorFormat::Structured {
                            dynamic_offset: false,
                        },
                    },
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
                    ty: pso::DescriptorType::Buffer {
                        ty: pso::BufferDescriptorType::Storage { read_only: false },
                        format: pso::BufferDescriptorFormat::Structured {
                            dynamic_offset: false,
                        },
                    },
                    count: 1,
                }],
                pso::DescriptorPoolCreateFlags::empty(),
            )
        }
        .expect("Can't create descriptor pool");
        (pipeline_layout, pipeline, set_layout, desc_pool)
    };

    let stride = std::mem::size_of::<u32>() as u64;
    let (staging_memory, staging_buffer, staging_size) = unsafe {
        create_buffer::<B::Backend>(
            &device,
            &memory_properties.memory_types,
            memory::Properties::CPU_VISIBLE | memory::Properties::COHERENT,
            buffer::Usage::TRANSFER_SRC | buffer::Usage::TRANSFER_DST,
            stride,
            flat_u32_bms.len() as u64,
        )
    };

    unsafe {
        let mapping = device.map_memory(&staging_memory, 0..staging_size).unwrap();

        ptr::copy_nonoverlapping(
            flat_u32_bms.as_ptr() as *const u8,
            mapping,
            flat_u32_bms.len() * stride as usize,
        );
        device.uinmap_memory(&staging_memory);
    }

    let (device_memory, device_buffer, _device_buffer_size) = unsafe {
        create_buffer::<B::Backend>(
            &device,
            &memory_properties.memory_types,
            memory::Properties::DEVICE_LOCAL,
            buffer::Usage::TRANSFER_SRC | buffer::Usage::TRANSFER_DST | buffer::Usage::STORAGE,
            stride,
            flat_u32_bms.len() as u64,
        )
    };

    let desc_set;

    unsafe {
        desc_set = desc_pool.allocate_set(&set_layout).unwrap();
        device.write_descriptor_sets(Some(pso::DescriptorSetWrite {
            set: &desc_set,
            binding: 0,
            array_offset: 0,
            descriptors: Some(pso::Descriptor::Buffer(&device_buffer, None..None)),
        }));
    };

    let mut command_pool =
        unsafe { device.create_command_pool(family.id(), pool::CommandPoolCreateFlags::empty()) }
            .expect("Can't create command pool");
    let fence = device.create_fence(false).unwrap();

    assert_eq!(task.num_bms % task.workgroup_size[0], 0);
    let num_dispatch_groups = task.num_bms / task.workgroup_size[0];
    let timing_qp = device.create_query_pool();
    for _ in 0..num_execs {
        unsafe {
            let mut command_buffer = command_pool.allocate_one(command::Level::Primary);
            command_buffer.begin_primary(command::CommandBufferFlags::ONE_TIME_SUBMIT);
            command_buffer.copy_buffer(
                &staging_buffer,
                &device_buffer,
                &[command::BufferCopy {
                    src: 0,
                    dst: 0,
                    size: stride * numbers.len() as u64,
                }],
            );
            command_buffer.pipeline_barrier(
                pso::PipelineStage::TRANSFER..pso::PipelineStage::COMPUTE_SHADER,
                memory::Dependencies::empty(),
                Some(memory::Barrier::Buffer {
                    states: buffer::Access::TRANSFER_WRITE
                        ..buffer::Access::SHADER_READ | buffer::Access::SHADER_WRITE,
                    families: None,
                    target: &device_buffer,
                    range: None..None,
                }),
            );
            command_buffer.bind_compute_pipeline(&pipeline);
            command_buffer.bind_compute_descriptor_sets(&pipeline_layout, 0, &[desc_set], &[]);
            command_buffer.dispatch([num_dispatch_groups, 1, 1]);
            command_buffer.pipeline_barrier(
                pso::PipelineStage::COMPUTE_SHADER..pso::PipelineStage::TRANSFER,
                memory::Dependencies::empty(),
                Some(memory::Barrier::Buffer {
                    states: buffer::Access::SHADER_READ | buffer::Access::SHADER_WRITE
                        ..buffer::Access::TRANSFER_READ,
                    families: None,
                    target: &device_buffer,
                    range: None..None,
                }),
            );
            command_buffer.copy_buffer(
                &device_buffer,
                &staging_buffer,
                &[command::BufferCopy {
                    src: 0,
                    dst: 0,
                    size: stride * bms.len() as u64,
                }],
            );
            command_buffer.finish();

            queue_group.queues[0].submit_without_semaphores(Some(&command_buffer), Some(&fence));

            device.wait_for_fence(&fence, !0).unwrap();
            command_pool.free(Some(command_buffer));
        }

        let result = unsafe {
            let mapping = device.map_memory(&staging_memory, 0..staging_size).unwrap();
            let r = Vec::<u32>::from(slice::from_raw_parts::<u32>(
                mapping as *const u8 as *const u32,
                32 * bms.len(),
            ));
            device.uinmap_memory(&staging_memory);
            r
        };

        assert_eq!(flat_u32_bms.len(), result.len());
        let mut result_bms: Vec<BitMatrix> = vec![];
        assert!((0..task.num_bms)
            .iter()
            .map(|i| BitMatrix::from_u32s(&result[(i * 32)..(i + 1) * 32])
                .unwrap()
                .identical_to(bms[i].transpose()))
            .all());
        println!("GPU results verified!");
        task.dispatch_times.push(dispatch_time);
    }

    unsafe {
        device.destroy_command_pool(command_pool);
        device.destroy_descriptor_pool(desc_pool);
        device.destroy_descriptor_set_layout(set_layout);
        device.destroy_shader_module(shader);
        device.destroy_buffer(device_buffer);
        device.destroy_buffer(staging_buffer);
        device.destroy_fence(fence);
        device.destroy_pipeline_layout(pipeline_layout);
        device.free_memory(device_memory);
        device.free_memory(staging_memory);
        device.destroy_compute_pipeline(pipeline);
    }

    vec![]
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
