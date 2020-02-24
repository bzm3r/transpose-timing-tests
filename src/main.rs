#[cfg(feature = "vk")]
extern crate gfx_backend_vulkan as Vulkan;

#[cfg(feature = "dx12")]
extern crate gfx_backend_dx12 as Dx12;

extern crate gfx_hal as hal;

mod bitmats;
mod gpu;
mod task;

use gpu::GpuTestEnv;
use task::{TaskGroupDefn, NumCpuExecs, NumGpuExecs, SubgroupSizeLog2};

fn main() {
    #[cfg(debug_assertions)]
    env_logger::init();

    #[cfg(feature = "vk")]
    {
        let mut test_env = GpuTestEnv::<Vulkan::Backend>::vulkan();
        // test_env.set_task_group(TaskGroupDefn::Threadgroup(NumCpuExecs(101), NumGpuExecs(1001)));
        // test_env.time_task_group();
        // test_env.save_results();
        //
        // test_env.set_task_group(TaskGroupDefn::Shuffle(NumCpuExecs(101), NumGpuExecs(1001), SubgroupSizeLog2(6)));
        // test_env.time_task_group();
        // test_env.save_results();

        test_env.set_task_group(TaskGroupDefn::HybridShuffle(NumCpuExecs(101), NumGpuExecs(5001)));
        test_env.time_task_group();
        test_env.save_results();
    }
}
