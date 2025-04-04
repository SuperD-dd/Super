use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct VID {
    /// ID
    pub id: String,
}

impl Default for VID {
    fn default() -> Self {
        VID {
            id: "".to_string(),
        }
    }
}