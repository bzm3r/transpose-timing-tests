extern crate gfx_backend_dx12 as dx12back;
extern crate gfx_backend_vulkan as vkback;
extern crate gfx_hal as hal;
extern crate gfx_auxil as auxil;
extern crate spirv_cross;

use std::{fs, ptr, slice, str::FromStr, path::Path};
use hal::{adapter::MemoryType, buffer, command, memory, pool, prelude::*, pso, Device};
use spirv_cross::{spirv::{Ast as SpirvAst, Module as SpirvMod}, hlsl, ErrorCode as SCError};

use winapi::um::{d3d12, d3dcommon};
use wio::com::ComPtr;

#[derive(Clone)]
pub struct Blob(pub ComPtr<d3dcommon::ID3DBlob>);

#[derive(Clone)]
pub struct ShaderByteCode {
    pub bytecode: d3d12::D3D12_SHADER_BYTECODE,
    blob: Option<Blob>,
}

type InstanceName = String;

pub enum Kernel {
    Dx12(Path),
    Vk(Path),
}

pub enum TestVariant {
    Dx12(InstanceName, Kernel),
    Vk(InstanceName, Kernel),
}

pub struct DispatchTime(f64);

pub fn get_timings(tv: TestVariant, bm: [u32; 32], num_execs: usize) -> Vec<DispatchTime> {
    match tv {
        TestVariant::Dx12(inm, k) => {
            run_tests::<dx12_back::Backend>(inm, k, input_bm, num_execs)
        }
        TestVariant::Vk(inm, k) => {
            run_tests::<vk::Backend>(inm, k, input_bm, num_execs)
        }
    }
}

pub fn run_tests<B: hal::Backend>(instance_name: String, kernel: Kernel, input_bm: [u32; 32], num_execs: usize) -> Vec<DispatchTime> {
    #[cfg(debug_assertions)]
        env_logger::init();

    let instance =
    B::Instance::create(&instance_name, 1).expect(format!("Failed to create {} instance!", &instance_name));

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

    let kernel_mod  = match test_kernel {
        Kernel::Dx12(p) => {
            create_SM6_shader_module::<B>(&device, p)
        },
        Kernel::Vk(p) => {
            create_SPIRV_shader_module(&device, p)
        }
    };

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
            module: &shader,
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
            input_bm.len() as u64,
        )
    };

    unsafe {
        let mapping = device
            .map_memory(&staging_memory, 0 .. staging_size)
            .unwrap();
        ptr::copy_nonoverlapping(
            input_bm.as_ptr() as *const u8,
            mapping,
            input_bm.len() * stride as usize,
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
            numbers.len() as u64,
        )
    };

    let desc_set;

    unsafe {
        desc_set = desc_pool.allocate_set(&set_layout).unwrap();
        device.write_descriptor_sets(Some(pso::DescriptorSetWrite {
            set: &desc_set,
            binding: 0,
            array_offset: 0,
            descriptors: Some(pso::Descriptor::Buffer(&device_buffer, None .. None)),
        }));
    };

    let mut command_pool =
        unsafe { device.create_command_pool(family.id(), pool::CommandPoolCreateFlags::empty()) }
            .expect("Can't create command pool");
    let fence = device.create_fence(false).unwrap();
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
                pso::PipelineStage::TRANSFER .. pso::PipelineStage::COMPUTE_SHADER,
                memory::Dependencies::empty(),
                Some(memory::Barrier::Buffer {
                    states: buffer::Access::TRANSFER_WRITE
                        .. buffer::Access::SHADER_READ | buffer::Access::SHADER_WRITE,
                    families: None,
                    target: &device_buffer,
                    range: None .. None,
                }),
            );
            command_buffer.bind_compute_pipeline(&pipeline);
            command_buffer.bind_compute_descriptor_sets(&pipeline_layout, 0, &[desc_set], &[]);
            command_buffer.dispatch([32, 32, 1]);
            command_buffer.pipeline_barrier(
                pso::PipelineStage::COMPUTE_SHADER .. pso::PipelineStage::TRANSFER,
                memory::Dependencies::empty(),
                Some(memory::Barrier::Buffer {
                    states: buffer::Access::SHADER_READ | buffer::Access::SHADER_WRITE
                        .. buffer::Access::TRANSFER_READ,
                    families: None,
                    target: &device_buffer,
                    range: None .. None,
                }),
            );
            command_buffer.copy_buffer(
                &device_buffer,
                &staging_buffer,
                &[command::BufferCopy {
                    src: 0,
                    dst: 0,
                    size: stride * input_bm.len() as u64,
                }],
            );
            command_buffer.finish();

            queue_group.queues[0].submit_without_semaphores(Some(&command_buffer), Some(&fence));

            device.wait_for_fence(&fence, !0).unwrap();
            command_pool.free(Some(command_buffer));
        }

        unsafe {
            let mapping = device
                .map_memory(&staging_memory, 0 .. staging_size)
                .unwrap();
            println!(
                "Times: {:?}",
                slice::from_raw_parts::<u32>(mapping as *const u8 as *const u32, numbers.len()),
            );
            device.uinmap_memory(&staging_memory);
        }
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


fn create_SM6_dxil(
    name: &str,
    source: &str,
    entry_point: &str,
) -> Vec<u8> {
    hassle_rs::compile_hlsl(name, source, entry_point, "cs_6_0", &["/Zi"], &[])
        .and_then(|shader| hassle_rs::validate_dxil(&shader))?
}

fn glsl_to_hlsl(path: Path) -> String {
    let glsl = fs::read_to_string(p).unwrap();
    let file = glsl_to_spirv::compile(&glsl, glsl_to_spirv::ShaderType::Compute).unwrap();
    let spirv_words: Vec<u32> = pso::read_spirv(file).expect("could not read SPIR-V");

    let spirv_mod = SpirvMod::from_words(&spirv_words);
    let mut ast = SpirvAst::<hlsl::Target>::parse(&spirv_mod)?;
    ast.compile()?
}

fn create_SM6_shader_module(path: Path) -> ShaderModule {
    let src = glsl_to_hlsl(path);
    let dxil_bytes = create_SM6_dxil("", &src, "main");

    let mut shader_blob_ptr: *mut d3dcommon::ID3DBlob = ptr::null_mut();

}
