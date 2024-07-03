use serde::{Serialize, Deserialize};

#[derive(Debug, PartialEq, Eq, Clone, Serialize, Deserialize)]
pub enum SecsDataType {
    /// List
    L = 0x00,
    /// 二进制
    B = 0x20,
    /// bool量
    BOOlEAN = 0x24,
    /// 单字节有符号
    I1 = 0x64,
    /// 单字节无符号
    U1 = 0xA4,
    /// 字符串
    ASCII = 0x40,
    /// 16位无符号
    U2 = 0xA8,
    /// 16位有符号
    I2 = 0x68,
    /// 32位无符号
    U4 = 0xB0,
    /// 32位有符号
    I4 = 0x70,
    /// 单精度浮点数
    F4 = 0x90,
    /// 双进度浮点数
    F8 = 0x80,
}

impl From<u8> for SecsDataType {
    fn from(value: u8) -> Self {
        match value {
            0x00 => SecsDataType::L,
            0x20 => SecsDataType::B,
            0x24 => SecsDataType::BOOlEAN,
            0x64 => SecsDataType::I1,
            0xA4 => SecsDataType::U1,
            0x40 => SecsDataType::ASCII,
            0xA8 => SecsDataType::U2,
            0x68 => SecsDataType::I2,
            0xB0 => SecsDataType::U4,
            0x70 => SecsDataType::I4,
            0x90 => SecsDataType::F4,
            0x80 => SecsDataType::F8,
            _ => unreachable!("Invalid SecsDataType value"),
        }
    }
}

impl Into<u8> for SecsDataType {
    fn into(self) -> u8 {
        match self {
            SecsDataType::L => 0x00,
            SecsDataType::B => 0x20,
            SecsDataType::BOOlEAN => 0x24,
            SecsDataType::I1 => 0x64,
            SecsDataType::U1 => 0xA4,
            SecsDataType::ASCII => 0x40,
            SecsDataType::U2 => 0xA8,
            SecsDataType::I2 => 0x68,
            SecsDataType::U4 => 0xB0,
            SecsDataType::I4 => 0x70,
            SecsDataType::F4 => 0x90,
            SecsDataType::F8 => 0x80,
        }
    }
}