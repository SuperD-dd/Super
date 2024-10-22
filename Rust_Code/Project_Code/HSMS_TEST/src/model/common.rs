pub mod common {
    /// 数值转字节并移除字节中前面为零的字节
    pub fn get_byte_remove_zero(value: i32) -> Vec<u8> {
        let data: [u8; 4] = value.to_be_bytes();
        let mut length = 0;
    
        for i in 0..data.len() {
            if data[i] == 0x00 {
                length += 1;
            } else {
                break;
            }
        }
    
        let result: Vec<u8> = data[length..].to_vec();
        result.iter().rev().cloned().collect()
    }

    /// 数值转换16位，删除前面无效0
    pub fn get_u16_remove_zero(data: &[u8]) -> u16 {
        u16::from_le_bytes([data[0], data[1]])
    }

    pub fn bytes_to_int(src: &[u8], offset: usize) -> i32 {
        let value = ((src[offset] as i32 & 0xFF) << 24)
            | ((src[offset + 1] as i32 & 0xFF) << 16)
            | ((src[offset + 2] as i32 & 0xFF) << 8)
            | (src[offset + 3] as i32 & 0xFF);
        value
    }

    /// 输入字节数组返回整数，忽略前面0的情况，字节数组不能超过4
    pub fn bytes_to_value(sr: &[u8]) -> u32 {
        let mut temp = 0;
        let str_temp = bytes_to_bin(sr);
        for (i, c) in str_temp.chars().rev().enumerate() {
            let d = 2u32.pow(i as u32);
            temp += d * c.to_digit(2).unwrap();
        }
        temp
    }

    pub fn bytes_to_bin(bytes_test: &[u8]) -> String {
        let mut str_result = String::new();
        for byte in bytes_test {
            let str_temp = format!("{:08b}", byte);
            str_result.push_str(&str_temp);
        }
        str_result
    }
}