use chrono::Local;
use dotenv::dotenv;
use nix::unistd;
use regex::Regex;
use std::fs::OpenOptions;
use std::io::{self, BufRead, Write};
use std::os::unix::io::{AsRawFd, FromRawFd};
use std::thread;
use std::time::Duration;

pub fn driver_log_level_monitor_task() {
    // 加载环境变量
    dotenv().ok();

    // 获取日志文件路径，默认为当前目录
    let log_dir = dotenv::var("DRIVER_LOG_PATH").unwrap_or_else(|_| ".".to_string());
    let task_log_path = format!("{}/log.txt", log_dir);

    // 打开日志文件，如果文件不存在则创建
    let log_file = match OpenOptions::new()
        .create(true)
        .write(true)
        .append(true)
        .open(&task_log_path)
    {
        Ok(file) => file,
        Err(err) => {
            eprintln!("Failed to open or create log file: {:?}", err);
            return;
        }
    };

    // 创建管道
    let (reader_fd, writer_fd) = unistd::pipe().expect("Failed to create pipe");

    // 将标准输出和错误输出重定向到管道写入端
    if unistd::dup2(writer_fd, libc::STDOUT_FILENO).is_err() {
        eprintln!("Failed to redirect stdout");
    }
    if unistd::dup2(writer_fd, libc::STDERR_FILENO).is_err() {
        eprintln!("Failed to redirect stderr");
    }

    // 关闭管道写入端
    unistd::close(writer_fd).expect("Failed to close writer_fd");

    // 创建一个线程来处理日志流
    let log_thread = thread::spawn(move || {
        // 打开日志文件写入
        let log_file = OpenOptions::new()
            .create(true)
            .write(true)
            .append(true)
            .open(task_log_path)
            .expect("Failed to open log file");

        let mut log_writer = io::BufWriter::new(log_file);

        // 将管道读取端包裹为文件描述符
        let reader = unsafe { std::fs::File::from_raw_fd(reader_fd) };
        let reader = io::BufReader::new(reader);

        // 定义日志过滤条件
        let info_regex = Regex::new(r"(?i)\binfo\b").expect("Invalid regex");
        let filter_keywords = vec!["fds", "lib"]; // 过滤的关键词

        for line in reader.lines() {
            match line {
                Ok(log_message) => {
                    // 过滤逻辑：仅过滤日志等级为 info 且包含特定关键词的日志
                    if info_regex.is_match(&log_message)
                        && filter_keywords.iter().any(|kw| log_message.contains(kw))
                    {
                        continue; // 过滤掉
                    }

                    // 为每条日志消息添加时间戳
                    let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
                    writeln!(log_writer, "[{}] {}", timestamp, log_message)
                        .expect("Failed to write log message");
                }
                Err(err) => eprintln!("Failed to read from pipe: {:?}", err),
            }
        }
    });

    // 模拟日志的循环输出
    let log_levels = vec!["INFO", "DEBUG", "ERROR"];
    let log_messages = vec![
        "Starting task",
        "fds initializing",
        "Internal process running",
        "lib dependency loading",
        "Task failed",
        "Task completed",
    ];

    loop {
        // 随机选择日志等级和消息进行输出
        for level in &log_levels {
            for message in &log_messages {
                println!("{} {}", level, message);
                thread::sleep(Duration::from_millis(500)); // 模拟日志生成间隔
            }
        }
    }

    // 等待日志线程完成
    if let Err(err) = log_thread.join() {
        eprintln!("Log thread panicked: {:?}", err);
    }
}

fn main() {
    driver_log_level_monitor_task();
}
