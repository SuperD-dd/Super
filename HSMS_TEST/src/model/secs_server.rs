use std::{collections::HashMap, sync::{Arc, Mutex}};

use super::{secs_ceid::CEID, secs_ecid::ECID, secs_rptid::RPTID, secs_svid::SVID};

#[derive(Debug, Default, Clone)]
pub struct Secs {
    pub svid_map: Arc<Mutex<HashMap<String, SVID>>>,
    pub rptid_map: Arc<Mutex<HashMap<String, RPTID>>>,
    pub ceid_map: Arc<Mutex<HashMap<String, CEID>>>,
    pub ecid_map: Arc<Mutex<HashMap<String, ECID>>>,
}