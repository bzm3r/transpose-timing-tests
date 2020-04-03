#[cfg(feature = "vk")]
extern crate gfx_backend_vulkan as Vulkan;

#[cfg(feature = "dx12")]
extern crate gfx_backend_dx12 as Dx12;

#[cfg(feature = "metal")]
extern crate gfx_backend_metal as Metal;

extern crate gfx_hal as hal;

mod bitmats;
mod file_utils;
mod gpu;
mod task;

use gpu::GpuTestEnv;
use task::{NumCpuExecs, NumGpuExecs, TaskGroupDefn};
use shaderc::ShaderKind::Task;
use crate::task::{TaskGroup, KernelType};

fn main() {
    #[cfg(debug_assertions)]
    env_logger::init();

    if !(cfg!(feature = "vk") | cfg!(feature = "dx12") | cfg!(feature = "metal")) {
        panic!("no backend loaded! `cargo run --features X`, where X is one of vk, dx12 or metal");
    }

    let num_cpu_execs = NumCpuExecs(101);
    let num_gpu_execs = NumGpuExecs(1001);

    #[cfg(feature = "vk")]
    let mut test_env = GpuTestEnv::<Vulkan::Backend>::vulkan();

    #[cfg(feature = "metal")]
    let mut test_env = GpuTestEnv::<Metal::Backend>::metal();

    test_env.set_task_group(
        TaskGroupDefn {
            num_cpu_execs,
            num_gpu_execs,
            kernel_type: KernelType::Threadgroup1d32,
        }
    );
    test_env.time_task_group();
    test_env.save_results();

    test_env.set_task_group(
        TaskGroupDefn {
            num_cpu_execs,
            num_gpu_execs,
            kernel_type: KernelType::Threadgroup1d8,
        }
    );
    test_env.time_task_group();
    test_env.save_results();

    test_env.set_task_group(
        TaskGroupDefn {
            num_cpu_execs,
            num_gpu_execs,
            kernel_type: KernelType::Shuffle8,
        }
    );
    test_env.time_task_group();
    test_env.save_results();

    test_env.set_task_group(
        TaskGroupDefn {
            num_cpu_execs,
            num_gpu_execs,
            kernel_type: KernelType::Shuffle32,
        }
    );
    test_env.time_task_group();
    test_env.save_results();

    test_env.set_task_group(
        TaskGroupDefn {
            num_cpu_execs,
            num_gpu_execs,
            kernel_type: KernelType::Ballot32,
        }
    );
    test_env.time_task_group();
    test_env.save_results();

    test_env.set_task_group(
        TaskGroupDefn {
            num_cpu_execs,
            num_gpu_execs,
            kernel_type: KernelType::HybridShuffle32,
        }
    );
    test_env.time_task_group();
    test_env.save_results();

    test_env.set_task_group(
        TaskGroupDefn {
            num_cpu_execs,
            num_gpu_execs,
            kernel_type: KernelType::HybridShuffleAdaptive32,
        }
    );
    test_env.time_task_group();
    test_env.save_results();

    // no need to run the 2D versions anymore
    // test_env.set_task_group(
    //     TaskGroupDefn {
    //         num_cpu_execs,
    //         num_gpu_execs,
    //         kernel_type: KernelType::Threadgroup2d32,
    //     }
    // );
    // test_env.time_task_group();
    // test_env.save_results();
    //
    // test_env.set_task_group(
    //     TaskGroupDefn {
    //         num_cpu_execs,
    //         num_gpu_execs,
    //         kernel_type: KernelType::Threadgroup2d8,
    //     }
    // );
    // test_env.time_task_group();
    // test_env.save_results();
}
