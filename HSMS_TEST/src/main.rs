// use std::collections::HashMap;
// use std::sync::{Arc, Mutex};
// use tokio::net::TcpListener;
// use tokio::io::{AsyncReadExt, AsyncWriteExt};
// use tokio::sync::broadcast;
// use tokio::time::{sleep, Duration};

// struct FruitStore {
//     subscribers: HashMap<String, Vec<tokio::net::TcpStream>>,
// }

// impl FruitStore {
//     fn new() -> Self {
//         Self {
//             subscribers: HashMap::new(),
//         }
//     }

//     fn subscribe(&mut self, fruit: &str, client: tokio::net::TcpStream) {
//         self.subscribers.entry(fruit.to_string()).or_insert_with(Vec::new).push(client);
//     }

//     async fn notify_subscribers(&self, fruit: &str, message: &str) {
//         if let Some(clients) = self.subscribers.get(fruit) {
//             for client in clients.into_iter() {
//                 // let mut client = client.try_clone().expect("Failed to clone TcpStream");
//                 let _ = client.write_all(message.as_bytes()).await;
//             }
//         }
//     }
// }

// #[tokio::main]
// async fn main() {
//     let listener = TcpListener::bind("127.0.0.1:8080").await.unwrap();
//     println!("服务端启动在：127.0.0.1:8080");

//     let fruit_store = Arc::new(Mutex::new(FruitStore::new()));

//     // 模拟水果数量变化和通知
//     let fruit_store_clone = Arc::clone(&fruit_store);
//     tokio::spawn(async move {
//         let fruits = vec!["apple", "banana", "orange"];
//         loop {
//             for fruit in &fruits {
//                 sleep(Duration::from_secs(15)).await;
//                 let message = format!("{}数量有变化", fruit);
//                 fruit_store_clone.lock().unwrap().notify_subscribers(fruit, &message).await;
//                 println!("已通知订阅{}的客户端", fruit);
//             }
//         }
//     });


//     loop {
//         match listener.accept().await {
//             Ok((mut socket, _addr)) => {
//                 let fruit_store = Arc::clone(&fruit_store);

//                 tokio::spawn(async move {
//                     let mut buffer = [0; 1024];
                    
//                     // 假设客户端发送消息格式为 "subscribe:apple"
//                     match socket.read(&mut buffer).await {
//                         Ok(n) => {
//                             let message = String::from_utf8_lossy(&buffer[..n]);
//                             if message.starts_with("subscribe:") {
//                                 let fruit = message.trim().split(':').nth(1).unwrap_or("");
//                                 fruit_store.lock().unwrap().subscribe(fruit, socket);
//                                 println!("客户端订阅了: {}", fruit);
//                             }
//                         },
//                         Err(_e) => {}
//                     }
//                 });
//             }
//             Err(e) => println!("无法接受连接: {}", e),
//         }
//     }
// }

use tokio::net::TcpListener;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[tokio::main]
async fn main() {
    let listener = TcpListener::bind("127.0.0.1:8080").await.unwrap();
    println!("服务端启动在：127.0.0.1:8080");

    loop {
        match listener.accept().await {
            Ok((mut socket, addr)) => {
                println!("客户端{}连接", addr);

                tokio::spawn(async move {
                    let mut buffer = vec![0; 1024]; // 创建一个足够大的buffer来存放客户端发送的数据
                    
                    match socket.read(&mut buffer).await {
                        Ok(n) => {
                            // 解析客户端发送的数据
                            println!("receive data:{:?}", buffer);
                            let data = String::from_utf8_lossy(&buffer[..n]);
                            println!("接收到来自{}的数据: {}", addr, data);

                            // 此处可以基于`data`进行进一步的处理
                            // 例如，解析参数和处理订阅逻辑
                        },
                        Err(e) => println!("读取来自客户端{}的数据时出错: {}", addr, e),
                    }
                    
                    // 假设服务端需要回复客户端确认信息
                    if let Err(e) = socket.write_all(b"11").await {
                        println!("发送确认消息到客户端{}失败: {}", addr, e);
                    }
                });
            },
            Err(e) => println!("无法接受连接: {}", e),
        }
    }
}