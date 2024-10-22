use std::collections::HashMap;

use crate::{
    enums::{secs_data_type::SecsDataType, secs_type::SecsType},
    model::{secs_body::SecsBody, secs_body_common::SecsBodyCommon, secs_header::SecsHeader},
};

pub fn s1f4(secs_header: Option<SecsHeader>, body_byte: &[u8]) -> Vec<u8> {
    let mut hashmap = HashMap::new();
    hashmap.insert(6, 5);



    
    let mut value_list: Vec<String> = Vec::new();
    let mut variable_index: Vec<u32> = Vec::new();
    let mut secs_body = SecsBody::new_root(SecsDataType::B, None);
    secs_body.set_body(body_byte);
    println!("new_secs_body:{:#?}", secs_body);
    if let Some(sub_secs_bodies) = secs_body.iter_sub_secs_body() {
        for item in sub_secs_bodies.iter() {
            // 处理每个元素（item）
            if let Some(message) = &item.message {
                if let Ok(int_value) = message.parse::<u32>() {
                    variable_index.push(int_value);
                }
            }
        }
    }
    println!("variable_index: {:?}", variable_index);
    for index in &variable_index {
        if let Some(value) = hashmap.get(index) {
            let value_string = value.to_string();
            value_list.push(value_string);
        }
    }

    println!("value_list: {:?}", value_list);

    let mut resbody = SecsBody::new_root(SecsDataType::L, None);
    for value in &value_list {
        resbody.add(SecsBody::new(
            SecsDataType::ASCII,
            Some(value.to_string()),
        ));
    }

    println!("{:#?}", resbody);
    let (_, data) = SecsBodyCommon::create_sces_message(
        1,
        4,
        secs_header.unwrap(),
        resbody,
    );
    println!("data:{:?}", data);
    data.to_vec()
}
