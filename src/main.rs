mod bitmats;
mod gpu;
mod task;

#[cfg(feature = "vk")]
extern crate gfx_backend_vulkan as Vulkan;

#[cfg(feature = "dx12")]
extern crate gfx_backend_dx12 as Dx12;

extern crate gfx_hal as hal;
use hal::Instance;

use gpu::time_task;
use task::{Task, BackendVariant, KernelType};

fn main() {
    #[cfg(debug_assertions)]
    env_logger::init();

    #[cfg(feature = "vk")]
    let vk_instance =
        Vulkan::Instance::create("vk-back", 1).expect(&format!("could not create Vulkan instance"));

    #[cfg(feature = "dx12")]
    let dx12_instance =
        Dx12::Instance::create("dx12-back", 1).expect(&format!("could not create DX12 instance"));

    let mut tasks = Vec::<Task>::new();
    tasks.push(Task {
        name: String::from("Vk-Shuffle-0"),
        device_name: String::new(),
        num_bms: 4096,
        workgroup_size: [64, 1],
        /// Should be an odd number.
        num_execs_gpu: 1001,
        /// Should be an odd number.
        num_execs_cpu: 101,
        kernel_type: KernelType::Shuffle,
        backend: BackendVariant::Vk,
        timestamp_query_times: vec![],
        instant_times: vec![],
    });

    tasks.push(Task {
        name: String::from("Vk-Threadgroup-0"),
        device_name: String::new(),
        num_bms: 4096,
        workgroup_size: [4, 32],
        /// Should be an odd number.
        num_execs_gpu: 1001,
        /// Should be an odd number.
        num_execs_cpu: 101,
        kernel_type: KernelType::Threadgroup,
        backend: BackendVariant::Vk,
        timestamp_query_times: vec![],
        instant_times: vec![],
    });

    for task in tasks.iter_mut() {
        match task.backend {
            #[cfg(feature = "vk")]
            BackendVariant::Vk => {
                time_task::<Vulkan::Backend>(&vk_instance, task);
            }
            #[cfg(feature = "dx12")]
            BackendVariant::Dx12 => match task.kernel_type {
                KernelType::Threadgroup => {
                    #[cfg(feature = "dx12")]
                    time_task::<Dx12::Backend>(&dx12_instance, task);
                }
                _ => panic!(
                    "DX12 backend can only execute handle threadgroup kernel variant at the moment"
                ),
            },
        }
        println!("{}", task);
    }
}
