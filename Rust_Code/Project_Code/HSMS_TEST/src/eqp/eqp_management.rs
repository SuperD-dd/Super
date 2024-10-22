use futures::executor::block_on;
use tokio::sync::mpsc::Sender;
use crate::{eqp::{s0f0::s0f0, s1f2::s1f2}, model::secs_header::SecsHeader};


pub fn eqp_manger(tx: Sender<Vec<u8>>, receive_byte: &Vec<u8>) {
    let header_byte = &receive_byte[..14];
    let mut body_byte: &[u8];
    if receive_byte.len() > 14 {
        body_byte = &receive_byte[14..];
    } else {
        body_byte = &[];
    }
    println!("header{:?}, body{:?}", header_byte, body_byte);
    let secs_header = SecsHeader::new(header_byte);
    println!("secs_header:{:#?}", secs_header);
    let header_request = secs_header.unwrap().to_string();
    println!("{:?}", header_request);

    let msg = match header_request.as_str() {
        "S0F0" => {
            s0f0(secs_header, body_byte)
        }
        "S1F1" => {
            s1f2(secs_header, body_byte)
        }
        // "S1F3" => {
        //     S1F4()
        // }
        // "S2F41" => {
        //     S2F42()
        // }
        _ => unreachable!("Invalid SecsType value"),
    };
    let tx = tx.clone();
    let future = async move {
        println!("发送消息ing:{:?}", msg);
        tx.send(msg).await.unwrap();
    };
    // 使用阻塞运行时执行异步代码
    println!("准备发送消息");
    block_on(future);
    println!("发完了消息");
}