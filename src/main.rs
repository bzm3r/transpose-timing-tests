mod bitmats;
mod gpu;

#[cfg(feature = "vk")]
extern crate gfx_backend_vulkan as Vulkan;

#[cfg(feature = "dx12")]
extern crate gfx_backend_dx12 as Dx12;

extern crate gfx_hal as hal;
use hal::Instance;

use gpu::time_task;
use std::fmt;
use std::fmt::Write;

#[derive(Clone)]
pub enum KernelType {
    Threadgroup,
    Ballot,
    Shuffle,
}

impl fmt::Display for KernelType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KernelType::Threadgroup => write!(f, "{}", "threadgroup"),
            KernelType::Ballot => write!(f, "{}", "ballot"),
            KernelType::Shuffle => write!(f, "{}", "shuffle"),
        }
    }
}

#[derive(Clone)]
pub enum BackendVariant {
    #[cfg(feature = "vk")]
    Vk,
    #[cfg(feature = "dx12")]
    Dx12,
    Empty,
}

impl fmt::Display for BackendVariant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            #[cfg(feature = "vk")]
            BackendVariant::Vk => write!(f, "{}", "vk"),
            #[cfg(feature = "dx12")]
            BackendVariant::Dx12 => write!(f, "{}", "dx12"),
            _ => write!(f, "{}", "empty"),
        }
    }
}

#[derive(Clone)]
pub struct Task {
    pub name: String,
    pub device_name: String,
    pub num_bms: u32,
    pub workgroup_size: [u32; 2],
    pub num_execs_gpu: u32,
    pub num_execs_cpu: u32,
    pub kernel_type: KernelType,
    pub backend: BackendVariant,
    pub instant_times: Vec<f64>,
    pub timestamp_query_times: Vec<f64>,
}

impl Task {
    pub fn timestamp_time_stats(&self) -> (usize, f64, f64) {
        let avg_time = self.timestamp_query_times.iter().sum::<f64>()
            / (self.timestamp_query_times.len() as f64);
        let std_time = (self
            .timestamp_query_times
            .iter()
            .map(|t| (t - avg_time).powf(2.0))
            .sum::<f64>()
            / (self.timestamp_query_times.len() as f64))
            .powf(0.5);
        (self.timestamp_query_times.len(), avg_time, std_time)
    }

    pub fn instant_time_stats(&self) -> (usize, f64, f64) {
        let avg_time = self.instant_times.iter().sum::<f64>() / (self.instant_times.len() as f64);
        let std_time = (self
            .instant_times
            .iter()
            .map(|t| (t - avg_time).powf(2.0))
            .sum::<f64>()
            / (self.instant_times.len() as f64))
            .powf(0.5);
        (self.instant_times.len(), avg_time, std_time)
    }
}

impl fmt::Display for Task {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (ts_n, ts_avg, ts_std) = self.timestamp_time_stats();
        let (its_n, its_avg, its_std) = self.instant_time_stats();
        let mut s = String::new();
        write!(s, "\ntask name:{}\n", self.name).unwrap();
        write!(s, "device: {}\n", self.device_name).unwrap();
        write!(
            s,
            "num BMs: {}, TG size: {}\n",
            self.num_bms,
            self.workgroup_size[0] * self.workgroup_size[1]
        )
            .unwrap();
        write!(
            s,
            "CPU loops: {}, GPU loops: {}\n",
            self.num_execs_cpu, self.num_execs_gpu
        )
            .unwrap();
        write!(
            s,
            "timestamp stats (N = {}): {:.2} +/- {:.2} ms\n",
            ts_n, ts_avg, ts_std
        )
            .unwrap();
        write!(
            s,
            "instant stats (N = {}): {:.2} +/- {:.2} ms",
            its_n, its_avg, its_std
        )
            .unwrap();
        write!(f, "{}", s)
    }
}
//
// pub enum CreatedInstance {
//     #[cfg(feature = "vk")]
//     Vulkan(Vulkan::Instance),
//     #[cfg(feature = "dx12")]
//     Dx12(Dx12::Instance),
// }

fn main() {
    #[cfg(feature = "vk")]
    let vk_instance = Vulkan::Instance::create("vk-back", 1)
        .expect(&format!("could not create Vulkan instance"));

    #[cfg(feature = "dx12")]
    let dx12_instance = Dx12::Instance::create("dx12-back", 1)
        .expect(&format!("could not create DX12 instance"));

    let mut tasks = Vec::<Task>::new();
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
            },
            #[cfg(feature = "dx12")]
            BackendVariant::Dx12 => {
                match task.kernel_type {
                    KernelType::Threadgroup => {
                        #[cfg(feature = "dx12")]
                        time_task::<Dx12::Backend>(&dx12_instance, task);
                    },
                    _ => {
                        panic!("DX12 backend can only execute handle threadgroup kernel variant at the moment")
                    }
                }
            },
            _ => {
                println!("Empty backend specified for task. Doing nothing.")
            }
        }
        println!("{}", task);
    }
}
