use crate::{enums::secs_type::SecsType, model::secs_header::SecsHeader};

pub fn s0f0(secs_header: Option<SecsHeader>, body_byte: &[u8]) -> Vec<u8> {
    let s_type: SecsType = secs_header.unwrap().get_secs_type();
    println!("s_type:{:?}", s_type);
    let mut response: [u8; 14] = [0; 14];
    response[3] = 10;
    response[4] = secs_header.unwrap().up_session_id;
    response[5] = secs_header.unwrap().lower_session_id;
    response[8] = secs_header.unwrap().p_type;
    response[10..].copy_from_slice(&secs_header.unwrap().system_bytes);
    match s_type {
        SecsType::SelectReq => {
            let response_s_type = SecsType::SelectRsp;
            let response_s_type_int: u8 = response_s_type.into();
            response[9] = response_s_type_int;
            println!("response SelectReq:{:?}", response);
            response.to_vec()
        }
        SecsType::LinkTestReq => {
            let response_s_type = SecsType::LinkTestRsp;
            let response_s_type_int: u8 = response_s_type.into();
            response[9] = response_s_type_int;
            println!("responseLinkTestReq :{:?}", response);
            response.to_vec()
        }
        _ => {
            println!("Received unknown SecsType");
            Vec::new()
        }
    }
}
