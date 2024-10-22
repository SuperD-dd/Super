use crate::{
    enums::{secs_data_type::SecsDataType, secs_type::SecsType},
    model::{secs_body::SecsBody, secs_body_common::SecsBodyCommon, secs_header::SecsHeader},
};

pub fn s1f2(secs_header: Option<SecsHeader>, body_byte: &[u8]) -> Vec<u8> {
    let mut secs_body = SecsBody::new_root(SecsDataType::B, None);
    secs_body.set_body(body_byte);
    println!("new_secs_body*********:{:#?}", secs_body);
    let software = "FlexSCADA".to_string();
    let version = "v1.6.0".to_string();
    let mut resbody = SecsBody::new_root(SecsDataType::L, None);
    resbody.add(SecsBody::new(SecsDataType::BOOlEAN, Some("1".to_string())));
    let mut mainbody = SecsBody::new(SecsDataType::L, None);
    mainbody.add(SecsBody::new(SecsDataType::ASCII, Some(software)));
    mainbody.add(SecsBody::new(SecsDataType::ASCII, Some(version)));
    resbody.add(mainbody);
    println!("[secs] S1F1 response_secs_body: {:#?}", resbody);
    // 将SecsBody解析为二进制
    let (_, data) = SecsBodyCommon::create_sces_message(1, 2, secs_header.unwrap(), resbody);
    println!("[secs] S1F2 send secs_data:{:?}", data);
    data.to_vec()
}
