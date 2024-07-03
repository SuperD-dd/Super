use crate::enums::secs_type::SecsType;
use crate::enums::secs_data_type::SecsDataType;
use crate::model::secs_body::SecsBody;
use crate::model::secs_header::SecsHeader;
use crate::model::secs_message::SecsMessage;
use crate::model::common::common;
use crate::model::secs_body_common::SecsBodyCommon;

/*pub mod EQP{
    pub fn S1F2(secs_header: SecsHeader, data: Vec<u8>) -> Vec<u8> {
        let mut secs_body = SecsBody::new_root(SecsDataType::L, None);
        let mut sub_body1 = SecsBody::new(SecsDataType::B, Some("\u{2}".to_string()));
        let mut sub_body2 = SecsBody::new(SecsDataType::B, Some("\u{1}".to_string()));

        let mut secs_body2 = SecsBody::new_root(SecsDataType::L, None);
        secs_body.add(sub_body1);
        secs_body.add(sub_body2);
    }
}*/