use serde::{Serialize, Deserialize};

#[derive(Debug, PartialEq, Eq, Clone, Serialize, Deserialize)]
pub enum MachineStatus {
    Illegal,
    OnLine,
    OnLineLocal,
    OnLineRemote,
    Offline,
}