use serde::{Serialize, Deserialize};

use crate::enums::secs_data_type::SecsDataType;

#[derive(Serialize, Deserialize, Debug)]
pub struct SVID {
    /// ID
    pub id: String,
    /// 名称
    pub sv_name: String,
    /// 值
    pub value: String,
    pub secs_data_type: SecsDataType,
    /// 单位
    pub units: String,
}

impl Default for SVID {
    fn default() -> Self {
        SVID {
            id: "".to_string(),
            sv_name: "".to_string(),
            value: "0".to_string(),
            secs_data_type: SecsDataType::ASCII,
            units: "pcs".to_string(),
        }
    }
}