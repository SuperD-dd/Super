use std::sync::Arc;
use tokio::sync::Mutex;
use chrono::NaiveDateTime;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc::{Sender, Receiver};

use crate::enums::secs_machine_status::MachineStatus;

#[derive(Debug, Clone)]
pub struct ClientInfo {
    pub tx: Sender<Vec<u8>>,
    pub status: MachineStatus,
    pub connect_time: NaiveDateTime,
    pub last_activity_time: NaiveDateTime,
}