mod bitmats;
mod gpu;

use gpu::{run_timing_tests, BackendVariant, KernelType, Task};

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
