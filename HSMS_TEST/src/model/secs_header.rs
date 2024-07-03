use crate::enums::secs_type::SecsType;
#[derive(Debug, Clone, Copy)]
pub struct SecsHeader {
    pub byte_length: i32,
    pub up_session_id: u8,
    pub lower_session_id: u8,
    pub stream: u8,
    pub function: u8,
    pub p_type: u8,
    pub s_type: u8,
    pub system_bytes: [u8; 4],
}

impl Default for SecsHeader {
    fn default() -> Self {
        SecsHeader {
            byte_length: 0,
            up_session_id: 0x00,
            lower_session_id: 0x00,
            stream: 0x81,
            function: 0x0D, 
            p_type: 0x00,
            s_type: 0x02,
            system_bytes: [0; 4],
        }
    }
}

impl SecsHeader {
    pub fn new(data: &[u8]) -> Option<SecsHeader> {
        if data.len() < 14 {
            return None;
        }

        let byte_length = i32::from_be_bytes([data[0], data[1], data[2], data[3]]);
        let up_session_id = data[4];
        let lower_session_id = data[5];
        let stream = data[6];
        let function = data[7];
        let p_type = data[8];
        let s_type = data[9];
        let mut system_bytes = [0u8; 4];
        system_bytes.copy_from_slice(&data[10..14]);

        Some(SecsHeader {
            byte_length,
            up_session_id,
            lower_session_id,
            stream,
            function,
            p_type,
            s_type,
            system_bytes,
        })
    }

    pub fn get_bytes(&self) -> [u8; 10] {
        let mut data = [0u8; 10];
        data[0] = self.up_session_id;
        data[1] = self.lower_session_id;
        data[2] = self.stream;
        data[3] = self.function;
        data[4] = self.p_type;
        data[5] = self.s_type;
        data[6] = self.system_bytes[0];
        data[7] = self.system_bytes[1];
        data[8] = self.system_bytes[2];
        data[9] = self.system_bytes[3];
        data
    }

    pub fn get_secs_type(&self) -> SecsType {
        self.s_type.into()
    }

    pub fn to_string(&self) -> String {
        let stream = if self.function % 2 == 0 {
            self.stream
        } else {
            self.stream - 0x80
        };
        format!("S{}F{}", stream, self.function)
    }
}
