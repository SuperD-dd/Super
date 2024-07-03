use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct ECID {
    // ID
    pub id: String,
    // 名称
    pub name: String,
    // 值
    pub value: String,
}

impl ECID {
    pub fn new() -> Self {
        ECID {
            id: "".to_string(),
            name: "".to_string(),
            value: "".to_string(),
        }
    }
}
