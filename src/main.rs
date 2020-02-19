mod bitmats;
mod gpu;

use gpu::{run_timing_tests, BackendVariant, KernelType, Task};

fn main() {
    let mut t0: gpu::Task = Task {
        name: String::from("Vk-Threadgroup-0"),
        num_bms: 4096,
        workgroup_size: [4, 32, 1],
        /// Should be an odd number.
        num_execs_gpu: 1001,
        /// Should be an odd number.
        num_execs_cpu: 101,
        kernel_type: KernelType::Threadgroup,
        backend: BackendVariant::Vk,
        timestamp_query_times: vec![],
        instant_times: vec![],
    };
    run_timing_tests(&mut t0);

    println!("{}", t0);
}
