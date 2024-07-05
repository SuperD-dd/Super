use byteorder::{BigEndian, ByteOrder, LittleEndian};
use enums::secs_data_type::SecsDataType;
use enums::secs_type::SecsType;
use hex::encode;
use model::secs_body::SecsBody;
use model::secs_body_common::SecsBodyCommon;
use model::secs_ceid::CEID;
use model::secs_client::ClientInfo;
use model::secs_ecid::ECID;
use model::secs_header::SecsHeader;
use model::secs_rptid::RPTID;
use model::secs_server::Secs;
use model::secs_svid::SVID;
use model::secs_vid::VID;
use model::SecsVO::{
    XMLCEIDs, XMLECIDs, XMLRPTIDRef, XMLRPTIDs, XMLSVIDRef, XMLSVIDs, XMLSecs, XMLCEID, XMLECID,
    XMLRPTID, XMLSVID,
};
use futures::executor::block_on;
use nom::bytes::complete::{tag, take};
use nom::IResult;
use quick_xml::se;
use serde_xml_rs::from_str;
use HSMS_TEST::eqp::eqp_management;
use std::collections::HashMap;
use std::collections::LinkedList;
use std::collections::VecDeque;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::Mutex;
use std::thread;
use tokio::io::{self, AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync;
use tokio::sync::mpsc;
use tokio::time::{sleep, timeout, Duration, Instant};
use HSMS_TEST::model::secs_server;

pub mod enums;
pub mod model;

const HEADER_SIZE: usize = 4;
const TIMEOUT_DURATION: Duration = Duration::from_secs(1);

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "0.0.0.0:8080".parse::<SocketAddr>()?;

    let listener = TcpListener::bind(addr).await?;

    println!("Listening on: {}", addr);

    let mut secs_server = Arc::new(Secs::default());
    while let Ok((mut stream, addr)) = listener.accept().await {
        let secs_server = secs_server.clone();
        println!("客户端{}连接", addr);
        tokio::spawn(async move {
            println!("1");
            if let Err(e) = handle_client(
                addr.clone(),
                secs_server,
                &mut stream,
            ).await {
                println!("Error handling client: {:?}", e);
            }
        });
    }

    Ok(())
}

async fn handle_client(
    addr: SocketAddr,
    secs_server: Arc<Secs>,
    stream: &mut TcpStream,
) -> Result<(), Box<dyn std::error::Error>> {
    let (tx, mut rx) = mpsc::channel(100);
    secs_server.add_client(addr, tx.clone());

    println!("222");
    let mut buffer = [0u8; 1024];
    let binary_list: Arc<Mutex<Vec<Vec<u8>>>> = Arc::new(Mutex::new(Vec::new()));
    let mut hashmap: HashMap<u32, u32> = HashMap::new();
    hashmap.insert(1000, 10);
    hashmap.insert(1001, 11);
    hashmap.insert(1002, 12);
    hashmap.insert(1003, 13);
    hashmap.insert(1004, 14);
    hashmap.insert(1005, 15);

    {
        let tx = tx.clone();
        let binary_list = Arc::clone(&binary_list);
        thread::spawn(move|| loop {
            {
                let mut binary_list_arc = binary_list.lock().unwrap();
                if binary_list_arc.len() > 0 {
                    for (i, receive_byte) in binary_list_arc.iter().enumerate() {
                        eqp_management::eqp_manger(tx.clone(), receive_byte);
                    //     println!("Row {}: {:?}", i, receive_byte);
                    //     let header_byte = &receive_byte[..14];
                    //     let mut body_byte: &[u8];
                    //     if receive_byte.len() > 14 {
                    //         body_byte = &receive_byte[14..];
                    //     } else {
                    //         body_byte = &[];
                    //     }
                    //     println!("header{:?}, body{:?}", header_byte, body_byte);
                    //     let secs_header: Option<SecsHeader> = SecsHeader::new(header_byte);
                    //     println!("secs_header:{:#?}", secs_header);
                    //     let header_request = secs_header.unwrap().to_string();
                    //     println!("{:?}", header_request);
    
                    //     let msg = match header_request.as_str() {
                    //         "S0F0" => {
                    //             let s_type: SecsType = secs_header.unwrap().get_secs_type();
                    //             println!("s_type:{:?}", s_type);
                    //             let mut response: [u8; 14] = [0; 14];
                    //             response[3] = 10;
                    //             response[4] = secs_header.unwrap().up_session_id;
                    //             response[5] = secs_header.unwrap().lower_session_id;
                    //             response[8] = secs_header.unwrap().p_type;
                    //             response[10..].copy_from_slice(&secs_header.unwrap().system_bytes);
                    //             match s_type {
                    //                 SecsType::SelectReq => {
                    //                     let response_s_type = SecsType::SelectRsp;
                    //                     let response_s_type_int: u8 = response_s_type.into();
                    //                     response[9] = response_s_type_int;
                    //                     println!("response SelectReq:{:?}", response);
                    //                     response.to_vec()
                    //                 }
                    //                 SecsType::LinkTestReq => {
                    //                     let response_s_type = SecsType::LinkTestRsp;
                    //                     let response_s_type_int: u8 = response_s_type.into();
                    //                     response[9] = response_s_type_int;
                    //                     println!("responseLinkTestReq :{:?}", response);
                    //                     response.to_vec()
                    //                 }
                    //                 _ => {
                    //                     println!("Received unknown SecsType");
                    //                     Vec::new()
                    //                 }
                    //             }
                    //         }
                    //         "S1F1" => {
                    //             let mut secs_body = SecsBody::new_root(SecsDataType::B, None);
                    //             secs_body.set_body(body_byte);
                    //             println!("new_secs_body*********:{:#?}", secs_body);
                    //             let software = "FlexSCADA".to_string();
                    //             let version = "v1.6.0".to_string();
                    //             let mut resbody = SecsBody::new_root(SecsDataType::L, None);
                    //             resbody
                    //                 .add(SecsBody::new(SecsDataType::BOOlEAN, Some("1".to_string())));
                    //             let mut mainbody = SecsBody::new(SecsDataType::L, None);
                    //             mainbody.add(SecsBody::new(SecsDataType::ASCII, Some(software)));
                    //             mainbody.add(SecsBody::new(SecsDataType::ASCII, Some(version)));
                    //             resbody.add(mainbody);
                    //             println!("[secs] S1F1 response_secs_body: {:#?}", resbody);
                    //             // 将SecsBody解析为二进制
                    //             let (_, data) = SecsBodyCommon::create_sces_message(
                    //                 1,
                    //                 2,
                    //                 secs_header.unwrap(),
                    //                 resbody,
                    //             );
                    //             println!("[secs] S1F2 send secs_data:{:?}", data);
                    //             data.to_vec()
                    //         }
                    //         "S1F3" => {
                    //             let mut value_list: Vec<String> = Vec::new();
                    //             let mut variable_index: Vec<u32> = Vec::new();
                    //             let mut secs_body = SecsBody::new_root(SecsDataType::B, None);
                    //             secs_body.set_body(body_byte);
                    //             println!("new_secs_body:{:#?}", secs_body);
                    //             if let Some(sub_secs_bodies) = secs_body.iter_sub_secs_body() {
                    //                 for item in sub_secs_bodies.iter() {
                    //                     // 处理每个元素（item）
                    //                     if let Some(message) = &item.message {
                    //                         if let Ok(int_value) = message.parse::<u32>() {
                    //                             variable_index.push(int_value);
                    //                         }
                    //                     }
                    //                 }
                    //             }
                    //             println!("variable_index: {:?}", variable_index);
                    //             for index in &variable_index {
                    //                 if let Some(value) = hashmap.get(index) {
                    //                     let value_string = value.to_string();
                    //                     value_list.push(value_string);
                    //                 }
                    //             }
    
                    //             println!("value_list: {:?}", value_list);
    
                    //             let mut resbody = SecsBody::new_root(SecsDataType::L, None);
                    //             for value in &value_list {
                    //                 resbody.add(SecsBody::new(
                    //                     SecsDataType::ASCII,
                    //                     Some(value.to_string()),
                    //                 ));
                    //             }
    
                    //             println!("{:#?}", resbody);
                    //             let (_, data) = SecsBodyCommon::create_sces_message(
                    //                 1,
                    //                 4,
                    //                 secs_header.unwrap(),
                    //                 resbody,
                    //             );
                    //             println!("data:{:?}", data);
                    //             data.to_vec()
                    //         }
                    //         "S2F41" => {
                    //             let mut value_list: Vec<String> = Vec::new();
                    //             let mut variable_index: Vec<u32> = Vec::new();
                    //             let mut secs_body = SecsBody::new_root(SecsDataType::B, None);
                    //             secs_body.set_body(body_byte);
                    //             println!("new_secs_body:{:#?}", secs_body);
                    //             if let Some(sub_secs_bodies) = secs_body.iter_sub_secs_body() {
                    //                 for item in sub_secs_bodies.iter() {
                    //                     // 处理每个元素（item）
                    //                     match item.data_type {
                    //                         SecsDataType::ASCII => {
                    //                             // 处理 ASCII 类型
                    //                             if let Some(message) = &item.message {
                    //                                 if message == "WRITE" {
                    //                                     continue;
                    //                                 } else {
                    //                                     // 跳出循环
                    //                                     println!("RCMD error: {:?}", message);
                    //                                     break;
                    //                                 }
                    //                             }
                    //                         }
                    //                         SecsDataType::L => {
                    //                             // 处理 L 类型
                    //                             if let Some(sub_secs_bodies1) =
                    //                                 item.iter_sub_secs_body()
                    //                             {
                    //                                 for item1 in sub_secs_bodies1.iter() {
                    //                                     if let Some(sub_secs_bodies2) =
                    //                                         item1.iter_sub_secs_body()
                    //                                     {
                    //                                         for item2 in sub_secs_bodies2.iter() {
                    //                                             if let Some(message) = &item2.message {
                    //                                                 if let Ok(int_value) =
                    //                                                     message.parse::<u32>()
                    //                                                 {
                    //                                                     variable_index.push(int_value);
                    //                                                 }
                    //                                             }
                    //                                         }
                    //                                     }
                    //                                 }
                    //                             }
                    //                         }
                    //                         _ => unreachable!("S2F41 Invalid SecsDataType value"),
                    //                     }
                    //                 }
                    //             }
                    //             if variable_index.len() > 0 && variable_index.len() % 2 == 0 {
                    //                 let mut iter = variable_index.iter();
                    //                 while let (Some(key), Some(value)) = (iter.next(), iter.next()) {
                    //                     hashmap.insert(*key, *value);
                    //                 }
    
                    //                 println!("{:?}", hashmap);
                    //             }
    
                    //             let mut resbody = SecsBody::new_root(SecsDataType::L, None);
                    //             resbody.add(SecsBody::new(
                    //                 SecsDataType::ASCII,
                    //                 Some("WRITE".to_string()),
                    //             ));
                    //             resbody.add(SecsBody::new(SecsDataType::B, Some("\u{0}".to_string())));
                    //             println!("{:#?}", resbody);
                    //             println!("variable_index:{:?}", variable_index);
                    //             let (_, data) = SecsBodyCommon::create_sces_message(
                    //                 1,
                    //                 4,
                    //                 secs_header.unwrap(),
                    //                 resbody,
                    //             );
                    //             println!("data:{:?}", data);
                    //             data.to_vec()
                    //         }
                    //         _ => unreachable!("Invalid SecsType value"),
                    //     };
                    //     let tx = tx.clone();
                    //     let future = async move {
                    //         println!("发送消息ing:{:?}", msg);
                    //         tx.send(msg).await.unwrap();
                    //     };
                    //     // 使用阻塞运行时执行异步代码
                    //     println!("准备发送消息");
                    //     block_on(future);
                    //     println!("发完了消息");
                    }
                    binary_list_arc.clear();
                }
                else {
                    thread::sleep(Duration::from_millis(50));
                }
            }    
        });
    }


    loop {
        tokio::select! {
            Some(msg) = rx.recv() => {
                println!("发送1111111");
                stream.write_all(&msg).await?;
                println!("发送22222");
            },
            result = stream.read(&mut buffer) => {
                match result {
                    Ok(n) if n == 0 => {
                        println!("Connection closed");
                        return Ok(());
                    }
                    Ok(n) => {
                        let length = BigEndian::read_u32(&buffer[..HEADER_SIZE]);
                        // let length = 20;
                        let control_request = buffer[..n].to_vec();
                        println!("data:{:?}, length = {}", control_request, length);
        
                        if length >= 10 {
                            println!("Okk10");
                            let mut binary_list = binary_list.lock().unwrap();
                            binary_list.push(control_request);
                            println!("Okk1011111");
                        } 
                        // else {
                        //     // Output start time for timeout
                        //     let start_time = Instant::now();
                        //     println!("Start waiting for data request at: {:?}", start_time);
        
                        //     let mut data_buf = vec![0; (length - 10) as usize];
        
                        //     tokio::select! {
                        //         result = stream.read_exact(&mut data_buf) => {
                        //             match result {
                        //                 Ok(_) => {
                        //                     let end_time = Instant::now();
                        //                     println!("Finished reading data request at: {:?}", end_time);
        
                        //                     let mut full_request = control_request;
                        //                     full_request.extend_from_slice(&data_buf);
        
                        //                     let mut binary_list = binary_list.lock().unwrap();
                        //                     binary_list.push(full_request);
                        //                 }
                        //                 Err(e) => {
                        //                     println!("Error reading data request: {:?}", e);
                        //                 }
                        //             }
                        //         }
                        //         _ = sleep(TIMEOUT_DURATION) => {
                        //             let end_time = Instant::now();
                        //             println!("Timed out waiting for data request at: {:?}", end_time-start_time);
                        //         }
                        //     }
                        // }
                    }
                    Err(e) => {
                        println!("Error reading from socket: {:?}", e);
                        return Err(Box::new(e));
                    }
                    _ => {
                        println!("Unexpected request size");
                        return Ok(());
                    }
                }
            },
        }
    }
}

async fn read_data_request(stream: &mut TcpStream, length: usize) -> io::Result<Vec<u8>> {
    let mut buf = vec![0; length];
    let read_start = Instant::now();
    println!("Start reading data request at: {:?}", read_start);
    stream.read_exact(&mut buf).await?;
    let read_end = Instant::now();
    println!("Finished reading data request at: {:?}", read_end);
    Ok(buf)
}

async fn handle_data_request(
    data: &[u8],
    request_id: &mut u8,
    request_list_tmp: &Arc<Mutex<Vec<Vec<u8>>>>,
    binary_list: &Arc<Mutex<Vec<Vec<u8>>>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut request_list = request_list_tmp.lock().unwrap();
    let request_id_usize = usize::from(*request_id - 1);

    if request_id_usize < request_list.len() {
        request_list[request_id_usize].extend_from_slice(data);

        let mut binary_list_arc = binary_list.lock().unwrap();
        binary_list_arc.push(request_list[request_id_usize].clone());
    } else {
        println!("Invalid request_id_usize: {}", request_id_usize);
    }

    Ok(())
}

/*
match stream.read(&mut buffer).await {
    Ok(n) if n == 0 => {
        println!("Connection closed");
        return Ok(());
    }
    Ok(n) => {
        println!("buffer: {:?}", &buffer[..n]);
        if is_data {
            handle_data_request(&buffer[..n], &mut request_id, &request_list_tmp, &binary_list).await?;
            is_data = false;
        } else {
            is_data = handle_control_request(&buffer[..n], &mut request_id, &request_list_tmp, &binary_list).await?;
        }
    }
    Err(e) => {
        println!("Error reading from socket: {:?}", e);
        return Err(Box::new(e));
    }
}
*/
async fn handle_control_request(
    data: &[u8],
    request_id: &mut u8,
    request_list_tmp: &Arc<Mutex<Vec<Vec<u8>>>>,
    binary_list: &Arc<Mutex<Vec<Vec<u8>>>>,
) -> Result<bool, Box<dyn std::error::Error>> {
    let mut request_list = request_list_tmp.lock().unwrap();
    *request_id = (request_list.len() + 1) as u8;
    request_list.push(data.to_vec());

    if data[3] > 10 {
        Ok(true)
    } else {
        let mut binary_list_arc = binary_list.lock().unwrap();
        binary_list_arc.push(data.to_vec());
        Ok(false)
    }
}

/*
fn parse_xml_to_secs(xml_data: &str) -> Secs {
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
            .map(|rptids| rptids.into_iter().map(|rptid_ref| VID { id: rptid_ref.id }).collect())
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

    Secs {
        svid_map: Arc::new(Mutex::new(svid_list)),
        rptid_map: Arc::new(Mutex::new(rptid_list)),
        ceid_map: Arc::new(Mutex::new(ceid_list)),
        ecid_map: Arc::new(Mutex::new(ecid_list)),
    }
}

fn generate_xml_from_secs(secs: &Secs) -> Result<String, Box<dyn std::error::Error>> {
    let svid_list = secs.svid_map.lock().unwrap();
    let ecid_list = secs.ecid_map.lock().unwrap();
    let rptid_list = secs.rptid_map.lock().unwrap();
    let ceid_list = secs.ceid_map.lock().unwrap();

    // 将 RPTID 和 CEID 中的字段转换为 XMLVIDRef 和 XMLRPTIDRef
    let xml_rptids: Vec<XMLRPTID> = rptid_list.values().map(|rptid| XMLRPTID {
        id: rptid.id.clone(),
        name: rptid.name.clone(),
        vid: rptid.vid_list.iter().map(|vid| XMLSVIDRef { id: vid.id.clone() }).collect(),
    }).collect();

    let xml_ceids: Vec<XMLCEID> = ceid_list.values().map(|ceid| XMLCEID {
        id: ceid.id.clone(),
        name: ceid.name.clone(),
        enable: ceid.enable,
        rptid: Some(ceid.rptid_list.iter().map(|rptid| XMLRPTIDRef { id: rptid.id.clone() }).collect()),
    }).collect();

    let xml_secs = XMLSecs {
        svids: XMLSVIDs {
            svid: svid_list.values().map(|svid| XMLSVID {
                id: svid.id.clone(),
                sv_name: svid.sv_name.clone(),
                values: svid.value.clone(),
                units: svid.units.clone(),
            }).collect(),
        },
        ecids: XMLECIDs {
            ecid: ecid_list.values().map(|ecid| XMLECID {
                id: ecid.id.clone(),
                name: ecid.name.clone(),
                value: ecid.value.clone(),
            }).collect(),
        },
        rptids: XMLRPTIDs {
            rptid: xml_rptids,
        },
        ceids: XMLCEIDs {
            ceid: xml_ceids,
        },
    };

    let xml_string = se::to_string(&xml_secs)?;
    Ok(xml_string)
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use std::collections::HashMap;


    #[test]
    fn test_parse_xml_to_secs1() {
        let simple_xml = r#"
            <Secs>
                <SVID>
                    <SVID ID="1000" SVName="STATUS" VALUES="0" UNITS="pcs"/>
                    <SVID ID="1005" SVName="IDLE" VALUES="0" UNITS="pcs"/>
                    <SVID ID="1006" SVName="BARCODE" VALUES="0" UNITS="pcs"/>
                    <SVID ID="1007" SVName="LOT_QTY" VALUES="0" UNITS="pcs"/>
                    <SVID ID="1008" SVName="TOTAL_QTY" VALUES="0" UNITS="pcs"/>
                    <SVID ID="1009" SVName="FLIP_KIT" VALUES="0" UNITS="pcs"/>
                    <SVID ID="1010" SVName="LOT_INFO" VALUES="0" UNITS="pcs"/>
                    <SVID ID="1011" SVName="REMOTE_STATUS" VALUES="0" UNITS="pcs"/>
                    <SVID ID="1012" SVName="DOOR" VALUES="0" UNITS="pcs"/>
                    <SVID ID="1013" SVName="UnitData" VALUES="0" UNITS="pcs"/>
                    <SVID ID="1014" SVName="UnitData" VALUES="0" UNITS="pcs"/>
                </SVID>
                <ECID>
                    <ECID ID="2000" Name="MACHINE_NAME" Value="STB-01"/>
                </ECID>
                <RPTID>
                    <RPTID ID="4000" Name="RP_MachineStatus">
                        <SVID ID="1000"/>
                    </RPTID>
                    <RPTID ID="4001" Name="RP_ReaderCode">
                        <SVID ID="1006"/>
                    </RPTID>
                    <RPTID ID="4002" Name="RP_Door">
                        <SVID ID="1012"/>
                    </RPTID>
                    <RPTID ID="4003" Name="RP_UnitData">
                        <SVID ID="1013"/>
                    </RPTID>
                    <RPTID ID="4004" Name="RP_UnitData">
                        <SVID ID="1014"/>
                    </RPTID>
                </RPTID>
                <CEID>
                    <CEID ID="3000" Name="CE_MachineStatus" Enable="true">
                        <RPTID ID="4000"/>
                    </CEID>
                    <CEID ID="3002" Name="CE_LotStart" Enable="true"/>
                    <CEID ID="3003" Name="CE_LotEnd" Enable="true"/>
                    <CEID ID="3004" Name="CE_DoorStatus" Enable="true">
                        <RPTID ID="4002"/>
                    </CEID>
                    <CEID ID="3005" Name="CE_ReaderTigger" Enable="true">
                        <RPTID ID="4001"/>
                    </CEID>
                    <CEID ID="3006" Name="CE_UnitDataUpload" Enable="true">
                        <RPTID ID="4003"/>
                    </CEID>
                    <CEID ID="3100" Name="CE_UnitDataUpload" Enable="true">
                        <RPTID ID="4004"/>
                    </CEID>
                </CEID>
            </Secs>
        "#;

        let secs = parse_xml_to_secs(simple_xml);

    }

    #[test]
    fn test_parse_xml_to_secs() {
        let xml_data = r#"
            <Secs>
                <SVID>
                    <SVID ID="1000" SVName="STATUS" VALUES="0" UNITS="pcs"/>
                    <SVID ID="1005" SVName="IDLE" VALUES="0" UNITS="pcs"/>
                    <SVID ID="1006" SVName="BARCODE" VALUES="0" UNITS="pcs"/>
                    <SVID ID="1007" SVName="LOT_QTY" VALUES="0" UNITS="pcs"/>
                    <SVID ID="1008" SVName="TOTAL_QTY" VALUES="0" UNITS="pcs"/>
                    <SVID ID="1009" SVName="FLIP_KIT" VALUES="0" UNITS="pcs"/>
                    <SVID ID="1010" SVName="LOT_INFO" VALUES="0" UNITS="pcs"/>
                    <SVID ID="1011" SVName="REMOTE_STATUS" VALUES="0" UNITS="pcs"/>
                    <SVID ID="1012" SVName="DOOR" VALUES="0" UNITS="pcs"/>
                    <SVID ID="1013" SVName="UnitData" VALUES="0" UNITS="pcs"/>
                    <SVID ID="1014" SVName="UnitData" VALUES="0" UNITS="pcs"/>
                </SVID>
                <ECID>
                    <ECID ID="2000" Name="MACHINE_NAME" Value="STB-01"/>
                </ECID>
                <RPTID>
                    <RPTID ID="4000" Name="RP_MachineStatus">
                        <SVID ID="1000"/>
                    </RPTID>
                    <RPTID ID="4001" Name="RP_ReaderCode">
                        <SVID ID="1006"/>
                    </RPTID>
                    <RPTID ID="4002" Name="RP_Door">
                        <SVID ID="1012"/>
                    </RPTID>
                    <RPTID ID="4003" Name="RP_UnitData">
                        <SVID ID="1013"/>
                    </RPTID>
                    <RPTID ID="4004" Name="RP_UnitData">
                        <SVID ID="1014"/>
                    </RPTID>
                </RPTID>
                <CEID>
                    <CEID ID="3000" Name="CE_MachineStatus" Enable="true">
                        <RPTID ID="4000"/>
                    </CEID>
                    <CEID ID="3002" Name="CE_LotStart" Enable="true"/>
                    <CEID ID="3003" Name="CE_LotEnd" Enable="true"/>
                    <CEID ID="3004" Name="CE_DoorStatus" Enable="true">
                        <RPTID ID="4002"/>
                    </CEID>
                    <CEID ID="3005" Name="CE_ReaderTigger" Enable="true">
                        <RPTID ID="4001"/>
                    </CEID>
                    <CEID ID="3006" Name="CE_UnitDataUpload" Enable="true">
                        <RPTID ID="4003"/>
                    </CEID>
                    <CEID ID="3100" Name="CE_UnitDataUpload" Enable="true">
                        <RPTID ID="4004"/>
                    </CEID>
                </CEID>
            </Secs>
        "#;

        let secs = parse_xml_to_secs(xml_data);
        {
            // Verify SVID list
            let svid_list = secs.svid_map.lock().unwrap();
            println!("svid_list: {:#?}", svid_list);
            assert_eq!(svid_list.len(), 11);
            assert_eq!(svid_list.get("1000").unwrap().sv_name, "STATUS");
            assert_eq!(svid_list.get("1005").unwrap().sv_name, "IDLE");

            // Verify ECID list
            let ecid_list = secs.ecid_map.lock().unwrap();
            println!("ecid_list: {:#?}", ecid_list);
            assert_eq!(ecid_list.len(), 1);
            assert_eq!(ecid_list.get("2000").unwrap().name, "MACHINE_NAME");

            // Verify RPTID list
            let rptid_list = secs.rptid_map.lock().unwrap();
            println!("rptid_list: {:#?}", rptid_list);
            assert_eq!(rptid_list.len(), 5);
            assert_eq!(rptid_list.get("4000").unwrap().name, "RP_MachineStatus");
            assert_eq!(rptid_list.get("4000").unwrap().vid_list.len(), 1);
            assert_eq!(rptid_list.get("4000").unwrap().vid_list[0].id, "1000");

            // Verify CEID list
            let mut ceid_list = secs.ceid_map.lock().unwrap();
            println!("ceid_list: {:?}", ceid_list);
            assert_eq!(ceid_list.len(), 7);
            assert_eq!(ceid_list.get("3000").unwrap().name, "CE_MachineStatus");
            assert!(ceid_list.get("3000").unwrap().enable);
            assert_eq!(ceid_list.get("3000").unwrap().rptid_list.len(), 1);
            assert_eq!(ceid_list.get("3000").unwrap().rptid_list[0].id, "4000");
            // ceid_list.clear();
        }
        let ans = generate_xml_from_secs(&secs).unwrap();
        println!("ans: {:#?}", ans);
    }
}
*/
