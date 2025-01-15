use tokio::sync::watch;
use tokio::time::{interval, Duration};
use tokio::select;
use std::time::Instant;
use chrono::Local;

fn get_current_time() -> String {
    // 获取当前时间并格式化为 "小时:分钟:秒.毫秒" 格式
    Local::now().format("%H:%M:%S%.3f").to_string()
}

#[tokio::main]
async fn main() {
    println!("{} 开始", get_current_time());
    // 创建一个 `watch` 通道用于发送关闭信号
    let (tx, rx) = watch::channel(false);

    // 启动一个任务，模拟发送停止信号
    let tx_clone = tx.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_secs(30)).await;
        println!("{} 发送停止信号", get_current_time());
        tx_clone.send(true).unwrap();
    });

    // 启动另一个任务，定时接收 `tick()` 和监听停止信号
    let tx_clone = tx.clone();
    tokio::spawn(async move {
        let mut interval = interval(Duration::from_secs(5));
        let mut counter = 0;
        let start_time = Instant::now();
        let mut rx = tx_clone.subscribe();
        loop {
            select! {
                // 每当定时器触发
                _ = interval.tick() => {
                    // 这里模拟定时任务的处理逻辑
                    counter += 1;
                    println!("{} tick #{}", get_current_time(), counter);

                    // 模拟处理逻辑需要 1 秒钟
                    tokio::time::sleep(Duration::from_secs(10)).await;
                    
                    println!("{} 处理了 tick", get_current_time());
                },
                // 检查停止信号
                _ = rx.changed() => {
                    if *rx.borrow_and_update() {
                        println!("{} 收到停止信号，退出任务", get_current_time());
                        break;
                    }
                }
            }
        }

        let duration = start_time.elapsed();
        println!("{} 任务执行了 {:?} 后结束", get_current_time(), duration);
    });

    // 由于两个任务是并发执行的，我们在这里用 sleep 保持主任务的生命周期
    // 如果没有这个，main 函数会立即结束，导致任务未能完成
    tokio::time::sleep(Duration::from_secs(40)).await;
}
