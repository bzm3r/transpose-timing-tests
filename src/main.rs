mod bitmats;
mod gpu;
mod task;

#[cfg(feature = "vk")]
extern crate gfx_backend_vulkan as Vulkan;

#[cfg(feature = "dx12")]
extern crate gfx_backend_dx12 as Dx12;

#[cfg(feature = "metal")]
extern crate gfx_backend_metal;

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

    #[cfg(feature = "metal")]
    let metal_instance =
        gfx_backend_metal::Instance::create("metal-back", 1).expect(&format!("could not create Metal instance"));

    let mut test_tasks = task::generate_threadgroup_tasks(32);

    for task in test_tasks.iter_mut() {
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
            #[cfg(feature = "metal")]
            BackendVariant::Metal => {
                time_task::<gfx_backend_metal::Backend>(&metal_instance, task);
            }
        }
        println!("{}", task);
    }
}
