use serde::{Serialize, Deserialize};

use super::secs_vid::VID;

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct CEID {
    // 事件ID
    pub id: String,
    // 事件的名称
    pub name: String,
    // 是否允许发送
    pub enable: bool,
    // 事件关联报告列表
    pub rptid_list: Vec<VID>,
}

impl CEID {
    pub fn new() -> Self {
        CEID {
            id: "".to_string(),
            name: "".to_string(),
            enable: false, 
            rptid_list: Vec::new(),
        }
    }
}
