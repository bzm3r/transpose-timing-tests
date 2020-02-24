use std::fmt;
use std::fmt::Write;
use std::fs::OpenOptions;

#[allow(dead_code)]
#[derive(Clone)]
pub enum KernelType {
    Threadgroup,
    Ballot,
    Shuffle,
    HybridShuffle,
}

impl fmt::Display for KernelType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KernelType::Threadgroup => write!(f, "{}", "threadgroup"),
            KernelType::Ballot => write!(f, "{}", "ballot"),
            KernelType::Shuffle => write!(f, "{}", "shuffle"),
            KernelType::HybridShuffle => write!(f, "{}", "hybrid-shuffle"),
        }
    }
}

#[derive(Clone)]
pub struct Task {
    pub name: String,
    pub workgroup_size: [u32; 2],
    pub instant_times: Vec<f64>,
    pub timestamp_query_times: Vec<f64>,
    pub kernel_name: String,
    pub kernel_type: KernelType,
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

    pub fn compile_kernel(&self) -> std::fs::File {
        let kernel_fp = format!("kernels/spv/{}.spv", self.kernel_name);
        match OpenOptions::new().read(true).open(&kernel_fp) {
            Ok(f) => {
                println!("{} kernel already compiled...", self.kernel_name);
                f
            }
            Err(_) => {
                println!("compiling kernel {}...", self.kernel_name);
                let glsl = self.materialize_kernel();
                let mut compiler = shaderc::Compiler::new().unwrap();
                let mut options = shaderc::CompileOptions::new().unwrap();
                options.set_target_env(
                    shaderc::TargetEnv::Vulkan,
                    ((1 as u32) << 22) | ((1 as u32) << 12),
                );
                let artifact = compiler
                    .compile_into_spirv(
                        &glsl,
                        shaderc::ShaderKind::Compute,
                        &format!("{}.glsl", self.kernel_name),
                        "main",
                        Some(&options),
                    )
                    .unwrap();
                let mut compiled_kernel = OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create(true)
                    .open(&kernel_fp)
                    .unwrap();
                use std::io::Write;
                compiled_kernel.write_all(artifact.as_binary_u8()).unwrap();

                compiled_kernel
            }
        }
    }

    pub fn materialize_kernel(&self) -> String {
        let tp = format!(
            "kernels/templates/transpose-{}-template.comp",
            self.kernel_type
        );
        let mut kernel = std::fs::read_to_string(&tp)
            .expect(&format!("could not kernel template at path: {}", &tp));

        match self.kernel_type {
            KernelType::Threadgroup => {
                kernel = kernel.replace("~WG_SIZE~", &format!("{}", self.workgroup_size[0]));
            }
            _ => {
                if self.workgroup_size[1] > 1 {
                    panic!("does not make sense to have Y-dimension in workgroup size for subgroup kernels");
                }
                kernel = kernel.replace("~WG_SIZE~", &format!("{}", self.workgroup_size[0]));
            }
        }

        #[cfg(debug_assertions)]
        std::fs::write(format!("kernels/comp/{}.comp", self.kernel_name), &kernel).unwrap();

        kernel
    }

    pub fn delete_compiled_kernel(&self) {
        let kp = format!("kernels/spv/{}.spv", self.kernel_name);
        match std::fs::read(&kp) {
            Ok(_) => {
                std::fs::remove_file(&kp).expect("could not delete compiled kernel");
            }
            _ => {}
        };
    }
}

impl fmt::Display for Task {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (ts_n, ts_avg, ts_std) = self.timestamp_time_stats();
        let (its_n, its_avg, its_std) = self.instant_time_stats();
        let mut s = String::new();
        write!(s, "task name:{}\n", self.name).unwrap();
        write!(
            s,
            "TG size: {}\n",
            self.workgroup_size[0] * self.workgroup_size[1]
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
            "instant stats (N = {}): {:.2} +/- {:.2} ms\n",
            its_n, its_avg, its_std
        )
        .unwrap();
        write!(f, "{}", s)
    }
}

#[derive(Clone, Copy)]
pub struct NumCpuExecs(pub u32);
#[derive(Clone, Copy)]
pub struct NumGpuExecs(pub u32);
#[derive(Clone, Copy)]
pub struct SubgroupSizeLog2(pub u32);

pub enum TaskGroupDefn {
    Threadgroup(NumCpuExecs, NumGpuExecs),
    Shuffle(NumCpuExecs, NumGpuExecs, SubgroupSizeLog2),
    HybridShuffle(NumCpuExecs, NumGpuExecs),
}

pub struct TaskGroup {
    pub name: String,
    pub num_bms: u32,
    pub num_gpu_execs: NumGpuExecs,
    pub num_cpu_execs: NumCpuExecs,
    pub kernel_type: KernelType,
    pub tasks: Vec<Task>,
}

impl fmt::Display for TaskGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}\nkernel type: {}\ncpu_execs: {}, gpu_execs: {}",
            self.name, self.kernel_type, self.num_cpu_execs.0, self.num_gpu_execs.0
        )
    }
}
