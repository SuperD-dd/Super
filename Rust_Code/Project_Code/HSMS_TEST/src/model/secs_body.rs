//use core::slice::SlicePattern;
use std::collections::LinkedList;
use std::convert::TryInto;
use crate::enums::secs_data_type::SecsDataType;
use crate::model::common::common;
use byteorder::{ByteOrder, LittleEndian, BigEndian};

#[derive(Debug, PartialEq, Eq)]
pub struct SecsBody {
    is_root: bool,
    pub data_type: SecsDataType,
    num_length_bytes: u8,
    body_length_bytes: Vec<u8>,
    body_length: i32,
    body_count: i32,
    pub message: Option<String>,
    body_bytes: Option<Vec<u8>>,
    comment: Option<String>,
    sub_secs_body: Option<LinkedList<SecsBody>>,
}

impl SecsBody {
    pub fn new_root(data_type: SecsDataType, message: Option<String>) -> Self {
        SecsBody {
            is_root: true,
            data_type,
            num_length_bytes: 0,
            body_length_bytes: Vec::new(),
            body_length: 0,
            body_count: 0,
            message,
            body_bytes: None,
            comment: None,
            sub_secs_body: None,
        }
    }

    pub fn new(data_type: SecsDataType, message: Option<String>) -> Self {
        SecsBody {
            is_root: false,
            data_type,
            num_length_bytes: 0,
            body_length_bytes: Vec::new(),
            body_length: 0,
            body_count: 0,
            message,
            body_bytes: None,
            comment: None,
            sub_secs_body: None,
        }
    }

    pub fn add(&mut self, body: SecsBody) {
        if self.sub_secs_body.is_none() {
            self.sub_secs_body = Some(LinkedList::new());
        }
        let sub_secs_body = self.sub_secs_body.as_mut().unwrap();
        sub_secs_body.push_back(body);
        self.body_count += 1;
    }

    // ... (other methods)
    pub fn get_body(&mut self) -> Vec<u8> {
        let header: u8 = self.data_type.clone().into();

        self.body_count = 1;
        match self.data_type {
            SecsDataType::L => {
                if let Some(sub_secs_bodies) = &mut self.sub_secs_body {
                    let mut msg_byte: Vec<u8> = Vec::new();
                    for item in &mut *sub_secs_bodies {
                        msg_byte.extend(item.get_body());
                    }
                    self.body_length_bytes = common::get_byte_remove_zero(sub_secs_bodies.len().try_into().unwrap());
                    self.num_length_bytes = self.body_length_bytes.len() as u8;
                    self.body_bytes = Some(msg_byte);
                    self.body_count = sub_secs_bodies.len() as i32;
                } else {
                    self.body_length_bytes = vec![1];
                    self.num_length_bytes = 1;
                    self.body_count = 0;
                }
            },
            SecsDataType::B => {
                if let Some(message) = &self.message {
                    let msg_as_u8: Vec<u8> = message
                        .split(',')
                        .map(|s| s.trim())
                        .map(|s| s.trim_start_matches("0x"))
                        .filter_map(|s| u8::from_str_radix(s, 16).ok())
                        .collect();
                    let length = msg_as_u8.len() as i32;
                    self.body_bytes = Some(msg_as_u8);
                    self.body_length = length;
                    self.body_length_bytes = common::get_byte_remove_zero(length);
                    self.num_length_bytes = self.body_length_bytes.len() as u8;
                    println!("body_bytes:{:?}, len:{}", self.body_bytes, length);
                }
            },
            SecsDataType::BOOlEAN => {
                if let Some(message) = &self.message {
                    let msg_as_u8: Vec<u8> = message
                        .split(',')
                        .map(|s| s.parse::<u8>().unwrap_or_default())
                        .collect();
                    let length = msg_as_u8.len() as i32;
                    self.body_bytes = Some(msg_as_u8);
                    self.body_length = length;
                    self.body_length_bytes = common::get_byte_remove_zero(length);
                    self.num_length_bytes = self.body_length_bytes.len() as u8;
                    println!("body_bytes:{:?},len:{}", self.body_bytes, length);
                }
            },
            SecsDataType::I1 => {
                if let Some(message) = &self.message {
                    let msg_as_i8: Vec<i8> = message
                        .split(',')
                        .map(|s| s.parse::<i8>().unwrap_or_default())
                        .collect();
                    let msg_as_bytes: Vec<u8> = msg_as_i8.iter().map(|&num| num as u8).collect();
                    let length = msg_as_i8.len() as i32;
                    self.body_bytes = Some(msg_as_bytes);
                    self.body_length = length;
                    self.body_length_bytes = common::get_byte_remove_zero(length);
                    self.num_length_bytes = self.body_length_bytes.len() as u8;
                    println!("body_bytes:{:?},len:{}", self.body_bytes, length);
                }
            },
            SecsDataType::U1 => {
                if let Some(message) = &self.message {
                    let msg_as_u8: Vec<u8> = message
                        .split(',')
                        .map(|s| s.parse::<u8>().unwrap_or_default())
                        .collect();
                    let length = msg_as_u8.len() as i32;
                    self.body_bytes = Some(msg_as_u8);
                    self.body_length = length;
                    self.body_length_bytes = common::get_byte_remove_zero(length);
                    self.num_length_bytes = self.body_length_bytes.len() as u8;
                    println!("body_bytes:{:?},len:{}", self.body_bytes, length);
                }
            },
            SecsDataType::ASCII => {
                if let Some(message) = &self.message {
                    self.body_bytes = Some(message.as_bytes().to_vec());
                    self.body_length = message.len() as i32;
                    self.body_length_bytes = common::get_byte_remove_zero(self.body_bytes.as_ref().unwrap().len() as i32);
                    self.body_length_bytes.reverse();
                    self.num_length_bytes = self.body_length_bytes.len() as u8;
                    self.body_count = self.body_length;
                }
            },
            SecsDataType::U2 => {
                // U2 类型的处理
                if let Some(message) = &self.message {
                    let values = message.split(',')
                        .map(|s| s.parse::<u16>().unwrap())
                        .collect::<Vec<u16>>();
                    let mut buffer = Vec::new();
                    for value in values {
                        let mut temp_buf = [0u8; 2];
                        BigEndian::write_u16(&mut temp_buf, value);
                        buffer.extend_from_slice(&temp_buf);
                    }
                    let length: i32 = buffer.len() as i32;
                    self.body_bytes = Some(buffer);
                    self.body_length = length;
                    self.body_length_bytes = common::get_byte_remove_zero(length);
                    self.num_length_bytes = self.body_length_bytes.len() as u8;
                }
            },
            SecsDataType::I2 => {
                // I2 类型的处理
                if let Some(message) = &self.message {
                    let values = message.split(',')
                        .map(|s| s.parse::<i16>().unwrap())
                        .collect::<Vec<i16>>();
                    let mut buffer = Vec::new();
                    for value in values {
                        let mut temp_buf = [0u8; 2];
                        BigEndian::write_i16(&mut temp_buf, value);
                        buffer.extend_from_slice(&temp_buf);
                    }
                    let length = buffer.len() as i32;
                    self.body_bytes = Some(buffer);
                    self.body_length = length;
                    self.body_length_bytes = common::get_byte_remove_zero(length);
                    self.num_length_bytes = self.body_length_bytes.len() as u8;
                }
            },
            SecsDataType::U4 => {
                // U4 类型的处理
                if let Some(message) = &self.message {
                    let values = message.split(',')
                        .map(|s| s.parse::<u32>().unwrap())
                        .collect::<Vec<u32>>();
                    let mut buffer = Vec::new();
                    for value in values {
                        let mut temp_buf = [0u8; 4];
                        BigEndian::write_u32(&mut temp_buf, value);
                        buffer.extend_from_slice(&temp_buf);
                    }
                    let length = buffer.len() as i32;
                    self.body_bytes = Some(buffer);
                    self.body_length = length;
                    self.body_length_bytes = common::get_byte_remove_zero(length);
                    self.num_length_bytes = self.body_length_bytes.len() as u8;
                }
            },
            SecsDataType::I4 => {
                // I4 类型处理
                if let Some(message) = &self.message {
                    let values = message.split(',')
                        .map(|s| s.parse::<i32>().unwrap())
                        .collect::<Vec<i32>>();
                    let mut buffer = Vec::new();
                    for value in values {
                        let mut temp_buf = [0u8; 4];
                        BigEndian::write_i32(&mut temp_buf, value);
                        buffer.extend_from_slice(&temp_buf);
                    }
                    let length = buffer.len() as i32;
                    self.body_bytes = Some(buffer);
                    self.body_length = length;
                    self.body_length_bytes = common::get_byte_remove_zero(length);
                    self.num_length_bytes = self.body_length_bytes.len() as u8;
                }
            },
            SecsDataType::F4 => {
                // F4 浮点数处理
                if let Some(message) = &self.message {
                    let values = message.split(',')
                        .map(|s| s.parse::<f32>().unwrap())
                        .collect::<Vec<f32>>();
                    let mut buffer = Vec::new();
                    for value in values {
                        let value_bits = value.to_bits();
                        let mut temp_buf = [0u8; 4];
                        BigEndian::write_u32(&mut temp_buf, value_bits);
                        buffer.extend_from_slice(&temp_buf);
                    }
                    let length = buffer.len() as i32;
                    self.body_bytes = Some(buffer);
                    self.body_length = length;
                    self.body_length_bytes = common::get_byte_remove_zero(length);
                    self.num_length_bytes = self.body_length_bytes.len() as u8;
                }
            },
            SecsDataType::F8 => {
                // F8 双精度浮点数处理
                if let Some(message) = &self.message {
                    let values = message.split(',')
                        .map(|s| s.parse::<f64>().unwrap())
                        .collect::<Vec<f64>>();
                    let mut buffer = Vec::new();
                    for value in values {
                        let value_bits = value.to_bits();
                        let mut temp_buf = [0u8; 8];
                        BigEndian::write_u64(&mut temp_buf, value_bits);
                        buffer.extend_from_slice(&temp_buf);
                    }
                    let length = buffer.len() as i32;
                    self.body_bytes = Some(buffer);
                    self.body_length = length;
                    self.body_length_bytes = common::get_byte_remove_zero(length);
                    self.num_length_bytes = self.body_length_bytes.len() as u8;
                }
            },
        }
    
        let mut res: Vec<u8> = Vec::with_capacity(1 + self.body_length_bytes.len() + self.body_bytes.as_ref().unwrap_or(&Vec::new()).len());

        println!("header: {:?}",header);
        println!("num_length_bytes: {:?}",self.num_length_bytes);
        println!("body_length_bytes: {:?}",self.body_length_bytes);
        println!("body_bytes: {:?}",self.body_bytes);
        res.push(header + self.num_length_bytes);
        res.extend_from_slice(&self.body_length_bytes);
        if let Some(body_bytes) = &self.body_bytes {
            res.extend_from_slice(body_bytes);
        }
    
        res
    }

    pub fn iter_sub_secs_body(&self) -> Option<&LinkedList<SecsBody>> {
        self.sub_secs_body.as_ref()
    }

    pub fn set_body(&mut self, receive_bytes: &[u8]) {
        if receive_bytes.is_empty() {
            self.body_bytes = None;
            return;
        }
    
        let bin = format!("{:08b}", receive_bytes[0]);
        println!("bin = {:?}", bin);
        let header = u8::from_str_radix(&bin[0..6], 2).unwrap() << 2;
        let cnt = &bin[6..8];
        println!("header:{}, cnt = {}", header, cnt);
        self.num_length_bytes = u8::from_str_radix(cnt, 2).unwrap();
        self.body_length_bytes = receive_bytes[1..(1 + self.num_length_bytes as usize)].to_vec();
        self.body_length = common::bytes_to_value(&self.body_length_bytes) as i32;
        self.body_count = 1;
        self.data_type = SecsDataType::from(header as u8);
        //println!("data_type: {:?}",self.data_type );
        match self.data_type {
            SecsDataType::L => {
                self.body_count = self.body_length;
                if self.body_count >= 0 {
                    let mut body_bytes = vec![0u8; receive_bytes.len() - self.num_length_bytes as usize - 1];
                    body_bytes.copy_from_slice(&receive_bytes[(self.num_length_bytes as usize + 1)..]);
                    // println!("body_bytes: {:?}", body_bytes);
                    let mut index = 0;
                    let mut sub_secs_bodies = LinkedList::new();
                    for _i in 0..self.body_length{
                        let mut new_body  = SecsBody::new(SecsDataType::B, None); // Set proper data type
                        println!("index: {:?}    body_bytes{:?}    i:{:?}",index,body_bytes,_i);
                        let temp = &body_bytes[index..];
                        new_body .set_body(temp);
                        let mut boxed_new_body = Box::new(new_body);
                        index += boxed_new_body.get_body().len();
                        // println!("index:{:?}",index);
                        sub_secs_bodies.push_back( *boxed_new_body);
                    }
                    self.sub_secs_body = Some(sub_secs_bodies);
                }
            }
            SecsDataType::B => {
                let body_bytes = receive_bytes[(self.num_length_bytes as usize + 1)..
                                                        (self.num_length_bytes as usize + 1 + self.body_length as usize)].to_vec();                             
                self.body_bytes = Some(body_bytes.clone());
                println!("body_bytes222: {:?}", body_bytes);
                let results: Vec<String> = body_bytes.iter()
                    .map(|&byte| format!("0x{:02x}", byte))
                    .collect();
                // 将生成的16进制字符串vector用逗号连接成一个字符串
                self.message = Some(results.join(","));
                println!("message222: {:?}", self.message);
            }
            SecsDataType::BOOlEAN => {
                let body_bytes = receive_bytes[(self.num_length_bytes as usize + 1)..
                                                        (self.num_length_bytes as usize + 1 + self.body_length as usize)].to_vec();
                self.body_bytes = Some(body_bytes.clone());
                println!("body_bytes222: {:?}", body_bytes);
                let mut results: Vec<String> = Vec::new();
                for byte in body_bytes {
                    if byte != 0 {
                        results.push("1".to_string());
                    }
                    else{
                        results.push("0".to_string());
                    }
                }
                self.message = Some(results.join(","));
                println!("message222: {:?}", self.message);
            }
            SecsDataType::U1 => {
                let body_bytes = receive_bytes[(self.num_length_bytes as usize + 1)..
                                                        (self.num_length_bytes as usize + 1 + self.body_length as usize)].to_vec();
                self.body_bytes = Some(body_bytes.clone());
                println!("body_bytes222: {:?}", body_bytes);
                let mut results: Vec<String> = Vec::new();
                for byte in body_bytes {
                    results.push(byte.to_string());
                }
                self.message = Some(results.join(","));
                println!("message222: {:?}", self.message);
            }
            SecsDataType::I1 => {
                let body_bytes = receive_bytes[(self.num_length_bytes as usize + 1)..
                                                        (self.num_length_bytes as usize + 1 + self.body_length as usize)].to_vec();
                self.body_bytes = Some(body_bytes.clone());
                println!("body_bytes222: {:?}", body_bytes);
                let mut results: Vec<String> = Vec::new();
                for byte in body_bytes {
                    let signed_byte = byte as i8;
                    results.push(signed_byte.to_string());
                }
                self.message = Some(results.join(","));
                println!("message222: {:?}", self.message);
            }
            SecsDataType::ASCII => {
                let mut body_bytes = vec![0u8; self.body_length as usize];
                // println!("receive_bytes: {:?}",receive_bytes);
                // println!("num_length_bytes{:?}",self.num_length_bytes as usize + 1);
                body_bytes.copy_from_slice(&receive_bytes[(self.num_length_bytes as usize + 1)..
                                                                (self.num_length_bytes as usize + 1 + self.body_length as usize)]);
                self.body_bytes = Some(body_bytes.clone());
                self.body_count = self.body_length;
                self.message = Some(String::from_utf8_lossy(&body_bytes).to_string());
            }
            SecsDataType::U2 => {
                let body_bytes = receive_bytes[(self.num_length_bytes as usize + 1)..
                                                        (self.num_length_bytes as usize + 1 + self.body_length as usize)].to_vec();                                
                self.body_bytes = Some(body_bytes.clone());
                println!("body_bytes222: {:?}", body_bytes);
                let mut messages = Vec::new();
                let chunks = body_bytes.chunks_exact(2);
                for chunk in chunks {
                    let value = BigEndian::read_u16(chunk);
                    messages.push(value.to_string());
                }
                self.message = Some(messages.join(","));
                println!("message222: {:?}", self.message);
            }
            SecsDataType::I2 => {
                let body_bytes = receive_bytes[(self.num_length_bytes as usize + 1)..
                                                        (self.num_length_bytes as usize + 1 + self.body_length as usize)].to_vec();                                
                self.body_bytes = Some(body_bytes.clone());
                println!("body_bytes222: {:?}", body_bytes);
                let mut messages = Vec::new();
                let chunks = body_bytes.chunks_exact(2);
                for chunk in chunks {
                    let value = BigEndian::read_i16(chunk);
                    messages.push(value.to_string());
                }
                self.message = Some(messages.join(","));
                println!("message222: {:?}", self.message);
            }
            SecsDataType::U4 => {
                let body_bytes = receive_bytes[(self.num_length_bytes as usize + 1)..
                                                (self.num_length_bytes as usize + 1 + self.body_length as usize)].to_vec();    
                self.body_bytes = Some(body_bytes.clone());                                                       
                if body_bytes.len() >= 4 {
                    let data_length = body_bytes.len() / 4;
                    self.body_count = data_length as i32; 
                    let mut values = Vec::new();
                    // 使用chunks_exact来安全地迭代每4个字节
                    for chunk in body_bytes.chunks_exact(4) {
                        // 直接按照小端序读取u32值
                        let value = LittleEndian::read_u32(chunk);
                        values.push(value);
                    }
                    // 将u32值转换为String，并用逗号连接
                    let values_str = values.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(",");
                    self.message = Some(values_str);
                } else if body_bytes.len() == 0 {
                    // 如果长度为0，将message设置为None
                    self.body_count = 0;
                    self.message = None;
                }
            }
            SecsDataType::I4 => {
                let body_bytes = receive_bytes[(self.num_length_bytes as usize + 1)..
                                                        (self.num_length_bytes as usize + 1 + self.body_length as usize)].to_vec();                                
                self.body_bytes = Some(body_bytes.clone());
                println!("body_bytes222: {:?}", body_bytes);
                let mut messages = Vec::new();
                let chunks = body_bytes.chunks_exact(4);
                for chunk in chunks {
                    let value = BigEndian::read_i32(chunk);
                    messages.push(value.to_string());
                }
                self.message = Some(messages.join(","));
                println!("message222: {:?}", self.message);
            }
            SecsDataType::F4 => {
                let body_bytes = receive_bytes[(self.num_length_bytes as usize + 1)..
                                                        (self.num_length_bytes as usize + 1 + self.body_length as usize)].to_vec();                                
                self.body_bytes = Some(body_bytes.clone());
                println!("body_bytes222: {:?}", body_bytes);
                let mut messages = Vec::new();
                let chunks = body_bytes.chunks_exact(4);
                for chunk in chunks {
                    let value = BigEndian::read_f32(chunk);
                    messages.push(value.to_string());
                }
                self.message = Some(messages.join(","));
                println!("message222: {:?}", self.message);
            }
            SecsDataType::F8 => {
                let body_bytes = receive_bytes[(self.num_length_bytes as usize + 1)..
                                                        (self.num_length_bytes as usize + 1 + self.body_length as usize)].to_vec();                                
                self.body_bytes = Some(body_bytes.clone());
                println!("body_bytes222: {:?}", body_bytes);
                let mut messages = Vec::new();
                let chunks = body_bytes.chunks_exact(8);
                for chunk in chunks {
                    let value = BigEndian::read_f64(chunk);
                    messages.push(value.to_string());
                }
                self.message = Some(messages.join(","));
                println!("message222: {:?}", self.message);
            }
            // Handle other cases similarly
            // _ => unreachable!("Invalid SecsDataType value"),
        }
    }
    

}