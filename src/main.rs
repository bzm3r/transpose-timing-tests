mod gpu;


extern crate gfx_backend_dx12 as dx12_back;
extern crate gfx_backend_vulkan as vk_back;

use std::path::{Path, PathBuf};

fn main() {
    let shader_folder = Path::new("kernels");
    let tgk_path = shader_folder.join(Path::new("transpose-threadgroup.comp"));
    let bk_path = shader_folder.join(Path::new("transpose-ballot.comp"));
    let sk_path = shader_folder.join(Path::new("transpose-shuffle.comp"));

    let dx12_tgk = gpu::gen_dx12_kernel(&tgk_path);
    let dx12_bk = gpu::gen_dx12_kernel(&bk_path);

    let dx12_tgk_time = gpu::execute_test(dx12_tgk);
    let vk_tgk = gpu::gen_vk_kernel(&tgk_path);
    let vk_sk = gpu::gen_vk_kernel(&sk_path);
    let vk_bk = gpu::gen_vk_kernel(&bk_path);

    println!("Hello, world!");
}
