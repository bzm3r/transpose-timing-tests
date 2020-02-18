mod gpu;
mod bitmats;

extern crate gfx_backend_dx12 as dx12_back;
extern crate gfx_backend_vulkan as vk_back;

use std::path::{Path, PathBuf};
use gpu::{run_timing_tests, Task, KernelType, BackendVariant};

fn main() {
    let mut t0: gpu::Task = Task {
        name: String::from("Vk-Threadgroup-0"),
        num_bms: 4096,
        workgroup_size: [4, 32, 1],
        kernel_type: KernelType::Threadgroup,
        backend: BackendVariant::Vk,
        dispatch_times: vec![],
    };
    run_timing_tests(&mut t0, 1000);

    println!("{}", t0);
}
