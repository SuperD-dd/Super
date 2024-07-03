use tokio::net::TcpListener;
use tokio::io::{self, AsyncReadExt, AsyncWriteExt};
use tokio::time::{self, Duration};

#[tokio::main]
async fn main() -> io::Result<()> {
    let listener = TcpListener::bind("127.0.0.1:8080").await?;

    loop {
        let (mut socket, _) = listener.accept().await?;

        tokio::spawn(async move {
            let mut buf = [0u8; 1024];
            // 假设这是对控制请求的处理部分，并且我们已经读取了控制请求
            
            // 现在等待数据请求。使用tokio的timeout函数设置超时。
            match time::timeout(Duration::from_secs(3), socket.read(&mut buf)).await {
                Ok(result) => match result {
                    Ok(n) if n == 0 => {
                        println!("客户端关闭了连接");
                        return;
                    }
                    Ok(_) => {
                        // 在这里处理数据请求
                        println!("接收到数据请求");
                        // 发送回复或者其他处理...

                    }
                    Err(e) => {
                        eprintln!("读取请求发生错误: {:?}", e);
                        return;
                    },
                },
                Err(_) => {
                    // 超时处理
                    println!("数据请求超时。执行超时处理逻辑。");

                    // 在这里加入任何你需要的超时处理逻辑。例如，发送超时警告给客户端，或者关闭连接等。

                    // 示例：发送超时消息给客户端（注意这里使用send可能导致阻塞并建议异步发送）
                    if let Err(e) = socket.write_all(b"请求超时，请重试").await {
                        eprintln!("发送超时消息失败: {:?}", e);
                    }

                    // 也可以选择关闭socket
                    // drop(socket);
                }
            }
        });
    }
}