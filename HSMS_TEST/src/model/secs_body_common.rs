use crate::enums::secs_type::SecsType;
use crate::enums::secs_data_type::SecsDataType;
use crate::model::secs_body::SecsBody;
use crate::model::secs_header::SecsHeader;
use crate::model::secs_message::SecsMessage;
use crate::model::common::common;
pub struct SecsBodyCommon;

impl SecsBodyCommon {
    /// 创建一条消息内容
    pub fn create_sces_message(sn: u8, fm: u8, secs_header: SecsHeader, mut body: SecsBody) -> (SecsMessage,Vec<u8>) {
        let mut header = SecsHeader::default();
        header.up_session_id = secs_header.up_session_id;
        header.lower_session_id = secs_header.lower_session_id;
        header.stream = sn;
        header.function = fm;
        header.s_type = SecsType::Message.into();
        header.p_type = secs_header.p_type;
        
        let system_bytes = secs_header.system_bytes.to_vec();
        //system_bytes.reverse();
        println!("[SecsBodyCommon] reversed_system_bytes: {:?}", system_bytes);
        header.system_bytes[..4].copy_from_slice(&system_bytes[..4]);
        
        let header_bytes = header.get_bytes();
        println!("[SecsBodyCommon] header_data: {:?}", header_bytes);
        let body_bytes = body.get_body();
        println!("[SecsBodyCommon] body_data: {:?}", body_bytes);
        
        let data_len:usize = header_bytes.len() + body_bytes.len() + 4;
        let mut data = vec![0; data_len];
        
        let count_bytes = ((data_len - 4) as usize).to_le_bytes();
        let mut extended_array: [u8; 8] = [0; 8];
        println!("[SecsBodyCommon] count_bytes: {:?}", count_bytes);
        println!("[SecsBodyCommon] extended_array: {:?}", extended_array);
        extended_array[..4].copy_from_slice(&count_bytes[..4]);
        println!("[SecsBodyCommon] extended_array: {:?}", extended_array);
        extended_array[4..].copy_from_slice(&[0, 0, 0, 0]);
        let mut reversed_count_bytes = extended_array.clone();
        reversed_count_bytes.reverse();

        data[0..4].copy_from_slice(&reversed_count_bytes[4..8]);
        data[4..(4 + header_bytes.len())].copy_from_slice(&header_bytes);
        data[(4 + header_bytes.len())..].copy_from_slice(&body_bytes);
        println!("[SecsBodyCommon] response_data: {:?}", data);
        
        let is_request = (header.function >> 7) & 1 == 1;
        
        let message = SecsMessage::new (
            header,
            body,
            is_request,
        );
        
        (message,data)
    }
    

}
