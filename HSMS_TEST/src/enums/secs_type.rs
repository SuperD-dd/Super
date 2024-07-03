#[derive(Debug, PartialEq, Eq, Clone)]
pub enum SecsType {
    SelectReq = 0x01,
    SelectRsp = 0x02,
    DeSelectReq = 0x03,
    DeSelectRsp = 0x04,
    LinkTestReq = 0x05,
    LinkTestRsp = 0x06,
    SeperateReq = 0x09,
    RejectReq = 0x07,
    Message = 0x00,
}

impl From<u8> for SecsType {
    fn from(value: u8) -> Self {
        match value {
            0x01 => SecsType::SelectReq,
            0x02 => SecsType::SelectRsp,
            0x03 => SecsType::DeSelectReq,
            0x04 => SecsType::DeSelectRsp,
            0x05 => SecsType::LinkTestReq,
            0x06 => SecsType::LinkTestRsp,
            0x09 => SecsType::SeperateReq,
            0x07 => SecsType::RejectReq,
            _ => unreachable!("Invalid SecsType value"),
        }
    }
}

impl Into<u8> for SecsType {
    fn into(self) -> u8 {
        match self {
            SecsType::SelectReq => 0x01,
            SecsType::SelectRsp => 0x02,
            SecsType::DeSelectReq => 0x03,
            SecsType::DeSelectRsp => 0x04,
            SecsType::LinkTestReq => 0x05,
            SecsType::LinkTestRsp => 0x06,
            SecsType::SeperateReq => 0x09,
            SecsType::RejectReq => 0x07,
            SecsType::Message => 0x00,
        }
    }
}
