#[cfg(feature = "vk")]
extern crate gfx_backend_vulkan as Vulkan;

#[cfg(feature = "dx12")]
extern crate gfx_backend_dx12 as Dx12;

#[cfg(feature = "metal")]
extern crate gfx_backend_metal as Metal;

extern crate gfx_hal as hal;

mod bitmats;
mod gpu;
mod task;

use gpu::GpuTestEnv;
use task::{NumCpuExecs, NumGpuExecs, SubgroupSizeLog2, TaskGroupDefn};

fn main() {
    #[cfg(debug_assertions)]
    env_logger::init();

    if !(cfg!(feature = "vk") | cfg!(feature = "dx12") | cfg!(feature = "metal")) {
        panic!("no backend loaded! `cargo run --features X`, where X is one of vk, dx12 or metal");
    }

    let num_cpu_execs: u32 = 101;
    let num_gpu_execs: u32 = 1001;

    #[cfg(feature = "vk")]
    {
        let mut test_env = GpuTestEnv::<Vulkan::Backend>::vulkan();
        // test_env.set_task_group(TaskGroupDefn::Threadgroup(
        //     NumCpuExecs(num_cpu_execs),
        //     NumGpuExecs(num_gpu_execs),
        // ));
        // test_env.time_task_group();
        // test_env.save_results();
        //
        // test_env.set_task_group(TaskGroupDefn::Shuffle(
        //     NumCpuExecs(num_cpu_execs),
        //     NumGpuExecs(num_gpu_execs),
        //     SubgroupSizeLog2(6),
        // ));
        // test_env.time_task_group();
        // test_env.save_results();
        //
        // test_env.set_task_group(TaskGroupDefn::HybridShuffle(
        //     NumCpuExecs(num_cpu_execs),
        //     NumGpuExecs(num_gpu_execs),
        // ));
        // test_env.time_task_group();
        // test_env.save_results();
        test_env.set_task_group(TaskGroupDefn::Ballot(
            NumCpuExecs(num_cpu_execs),
            NumGpuExecs(num_gpu_execs),
            SubgroupSizeLog2(6),
        ));
        test_env.time_task_group();
        test_env.save_results();
    }

    #[cfg(feature = "metal")]
    {
        let mut test_env = GpuTestEnv::<Metal::Backend>::metal();
        test_env.set_task_group(TaskGroupDefn::Threadgroup(
            NumCpuExecs(num_cpu_execs),
            NumGpuExecs(num_gpu_execs),
        ));
        test_env.time_task_group();
        test_env.save_results();

        test_env.set_task_group(TaskGroupDefn::HybridShuffle(
            NumCpuExecs(num_cpu_execs),
            NumGpuExecs(num_gpu_execs),
        ));
        test_env.time_task_group();
        test_env.save_results();
    }
}
