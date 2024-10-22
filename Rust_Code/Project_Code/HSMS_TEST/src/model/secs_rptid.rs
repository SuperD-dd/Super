use serde::{Serialize, Deserialize};

use super::secs_vid::VID;

#[derive(Serialize, Deserialize, Debug, Default, Clone)]
pub struct RPTID {
    pub id: String,
    pub name: String,
    pub vid_list: Vec<VID>,
}

impl RPTID {
    pub fn new() -> Self {
        RPTID {
            id: "".to_string(),
            name: "".to_string(),
            vid_list: Vec::new(),
        }
    }
}
