use std::{
    collections::HashMap,
    net::SocketAddr,
    sync::{Arc, Mutex},
};

use crate::enums::{secs_data_type::SecsDataType, secs_machine_status::MachineStatus};
use chrono::Utc;
use quick_xml::se;
use serde_xml_rs::from_str;
use tokio::{
    io::AsyncWriteExt,
    net::{TcpListener, TcpStream},
};
use tokio::sync::mpsc::{Sender, Receiver};
use super::{
    secs_ceid::CEID,
    secs_client::ClientInfo,
    secs_ecid::ECID,
    secs_rptid::RPTID,
    secs_svid::SVID,
    secs_vid::VID,
    SecsVO::{
        XMLCEIDs, XMLECIDs, XMLRPTIDRef, XMLRPTIDs, XMLSVIDRef, XMLSVIDs, XMLSecs, XMLCEID,
        XMLECID, XMLRPTID, XMLSVID,
    },
};

#[derive(Debug, Default, Clone)]
pub struct Secs {
    pub peer: String,
    pub svid_map: Arc<Mutex<HashMap<String, SVID>>>,
    pub rptid_map: Arc<Mutex<HashMap<String, RPTID>>>,
    pub ceid_map: Arc<Mutex<HashMap<String, CEID>>>,
    pub ecid_map: Arc<Mutex<HashMap<String, ECID>>>,
    pub clients: Arc<Mutex<HashMap<SocketAddr, ClientInfo>>>,
}

impl Secs {
    pub fn add_client(&self, addr: SocketAddr, tx: Sender<Vec<u8>>,) {
        let connect_time = Utc::now().naive_utc();
        let last_activity_time = connect_time;
        let status = MachineStatus::default();
        let client_info = ClientInfo {
            status,
            connect_time,
            last_activity_time,
            tx: tx,
        };
        let mut clients = self.clients.lock().unwrap();
        clients.insert(addr, client_info);
    }

    pub async fn send_message(
        &self,
        addr: &SocketAddr,
        message: &Vec<u8>,
    ) -> Result<(), std::io::Error> {
        let clients = self.clients.lock().unwrap();
        if let Some(client_info) = clients.get(addr) {
            if client_info.tx.send(message.clone()).await.is_err() {
                println!("Receiver dropped");
            }
            return Ok(());
        }
        Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "Client not found or stream not available",
        ))
    }

    pub fn update_client_status(&self, addr: &SocketAddr, status: MachineStatus) {
        let mut clients = self.clients.lock().unwrap();
        if let Some(client_info) = clients.get_mut(addr) {
            client_info.status = status;
            client_info.last_activity_time = Utc::now().naive_utc();
        }
    }

    pub fn remove_client(&self, addr: &SocketAddr) {
        let mut clients = self.clients.lock().unwrap();
        clients.remove(addr);
    }

    pub fn update_last_activity(&self, addr: &SocketAddr) {
        let mut clients = self.clients.lock().unwrap();
        if let Some(client_info) = clients.get_mut(addr) {
            client_info.last_activity_time = Utc::now().naive_utc();
        }
    }

    pub fn parse_xml_to_secs(&mut self, xml_data: &str) {
        let xml_secs: XMLSecs = from_str(xml_data).expect("Failed to parse XML");

        let mut svid_list = HashMap::new();
        for svid in xml_secs.svids.svid {
            svid_list.insert(
                svid.id.clone(),
                SVID {
                    id: svid.id,
                    sv_name: svid.sv_name,
                    value: svid.values,
                    secs_data_type: SecsDataType::ASCII, // 假设SecsDataType为ASCII，可根据实际情况调整
                    units: svid.units,
                },
            );
        }

        let mut ecid_list = HashMap::new();
        for ecid in xml_secs.ecids.ecid {
            ecid_list.insert(
                ecid.id.clone(),
                ECID {
                    id: ecid.id,
                    name: ecid.name,
                    value: ecid.value,
                },
            );
        }

        let mut rptid_list = HashMap::new();
        for rptid in xml_secs.rptids.rptid {
            let svid_list: Vec<VID> = rptid
                .vid
                .into_iter()
                .map(|vid_ref| VID { id: vid_ref.id })
                .collect();
            rptid_list.insert(
                rptid.id.clone(),
                RPTID {
                    id: rptid.id,
                    name: rptid.name,
                    vid_list: svid_list,
                },
            );
        }

        let mut ceid_list = HashMap::new();
        for ceid in xml_secs.ceids.ceid {
            let rptid_list: Vec<VID> = ceid
                .rptid
                .map(|rptids| {
                    rptids
                        .into_iter()
                        .map(|rptid_ref| VID { id: rptid_ref.id })
                        .collect()
                })
                .unwrap_or_default();
            ceid_list.insert(
                ceid.id.clone(),
                CEID {
                    id: ceid.id,
                    name: ceid.name,
                    enable: ceid.enable,
                    rptid_list: rptid_list,
                },
            );
        }

        {
            let mut svid_map = self.svid_map.lock().unwrap();
            *svid_map = svid_list;
        }

        {
            let mut rptid_map = self.rptid_map.lock().unwrap();
            *rptid_map = rptid_list;
        }

        {
            let mut ceid_map = self.ceid_map.lock().unwrap();
            *ceid_map = ceid_list;
        }

        {
            let mut ecid_map = self.ecid_map.lock().unwrap();
            *ecid_map = ecid_list;
        }
    }

    pub fn generate_xml_from_secs(&self) -> Result<String, Box<dyn std::error::Error>> {
        let svid_list: HashMap<String, SVID>;
        let ecid_list: HashMap<String, ECID>;
        let rptid_list: HashMap<String, RPTID>;
        let ceid_list: HashMap<String, CEID>;
        {
            let svid_list_map_clone = self.svid_map.lock().unwrap();
            svid_list = svid_list_map_clone.clone();
        }
        {
            let ecid_list_map_clone = self.ecid_map.lock().unwrap();
            ecid_list = ecid_list_map_clone.clone();
        }
        {
            let rptid_list_map_clone = self.rptid_map.lock().unwrap();
            rptid_list = rptid_list_map_clone.clone();
        }
        {
            let ceid_list_map_clone = self.ceid_map.lock().unwrap();
            ceid_list = ceid_list_map_clone.clone();
        }

        // 将 RPTID 和 CEID 中的字段转换为 XMLVIDRef 和 XMLRPTIDRef
        let xml_rptids: Vec<XMLRPTID> = rptid_list
            .values()
            .map(|rptid| XMLRPTID {
                id: rptid.id.clone(),
                name: rptid.name.clone(),
                vid: rptid
                    .vid_list
                    .iter()
                    .map(|vid| XMLSVIDRef { id: vid.id.clone() })
                    .collect(),
            })
            .collect();

        let xml_ceids: Vec<XMLCEID> = ceid_list
            .values()
            .map(|ceid| XMLCEID {
                id: ceid.id.clone(),
                name: ceid.name.clone(),
                enable: ceid.enable,
                rptid: Some(
                    ceid.rptid_list
                        .iter()
                        .map(|rptid| XMLRPTIDRef {
                            id: rptid.id.clone(),
                        })
                        .collect(),
                ),
            })
            .collect();

        let xml_secs = XMLSecs {
            svids: XMLSVIDs {
                svid: svid_list
                    .values()
                    .map(|svid| XMLSVID {
                        id: svid.id.clone(),
                        sv_name: svid.sv_name.clone(),
                        values: svid.value.clone(),
                        units: svid.units.clone(),
                    })
                    .collect(),
            },
            ecids: XMLECIDs {
                ecid: ecid_list
                    .values()
                    .map(|ecid| XMLECID {
                        id: ecid.id.clone(),
                        name: ecid.name.clone(),
                        value: ecid.value.clone(),
                    })
                    .collect(),
            },
            rptids: XMLRPTIDs { rptid: xml_rptids },
            ceids: XMLCEIDs { ceid: xml_ceids },
        };

        let xml_string = se::to_string(&xml_secs)?;
        Ok(xml_string)
    }
}
