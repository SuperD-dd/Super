use fern::{Dispatch, log_file};
use std::{fs::OpenOptions, sync::{Arc, RwLock}};
use log::{LevelFilter, SetLoggerError};

#[derive(Clone)]
struct Logger {
    log_path: Arc<RwLock<String>>,
}

impl Logger {
    fn new(log_path: &str) -> Self {
        Logger {
            log_path: Arc::new(RwLock::new(log_path.to_string())),
        }
    }

    fn set_up_logger(&self) -> Result<(), SetLoggerError> {
        // 获取当前的日志路径
        let log_path = self.log_path.read().unwrap().clone();

        // 打开日志文件
        let log_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path).expect("111");

        // // 创建新的日志配置
        // let dispatch = Dispatch::new()
        //     .level(LevelFilter::Info)
        //     .chain(log_file);

        // 将日志配置转换为 log::Log 实现
        let (min_level, log) = fern::Dispatch::new()
            .level(log::LevelFilter::Info)
            .chain(log_file)
            .into_log();
        // 设置为当前的日志记录器
        log::set_boxed_logger(log)?;
        log::set_max_level(LevelFilter::Info);

        Ok(())
    }

    fn update_log_path(&self, new_path: &str) {
        let mut path = self.log_path.write().unwrap();
        *path = new_path.to_string();
    }
}

fn main() {
    let logger = Logger::new("log1.txt");

    // 初始化日志系统
    if let Err(e) = logger.set_up_logger() {
        eprintln!("Failed to initialize logger: {}", e);
        return;
    }
    log::info!("This is an info log with a new log path111");
    // 模拟一段时间后的日志路径更改
    std::thread::sleep(std::time::Duration::from_secs(2));

    // 更新日志路径
    logger.update_log_path("log2.txt");

    // 重新设置日志输出到新的文件路径
    if let Err(e) = logger.set_up_logger() {
        eprintln!("Failed to update logger: {}", e);
        return;
    }

    // 记录日志
    log::info!("This is an info log with a new log path");
}
