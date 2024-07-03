use std::error::Error;

#[derive(Debug, Default)]
pub struct IDGenerator {
    pub svid_start: usize,
    pub rptid_start: usize,
    pub ceid_start: usize,
    pub ecid_start: usize,
}

impl IDGenerator {
    pub fn new(len: usize) -> Self {
        let digits = ((len as f64).log10() + 1.0).floor() as u32;
        let basic_id = 10_usize.pow(digits);
        IDGenerator {
            svid_start: basic_id * 1,
            rptid_start: basic_id * 2,
            ceid_start: basic_id * 3,
            ecid_start: basic_id * 4,
        }
    }

    pub fn generate_id(&mut self, entity: &str) -> Result<String, Box<dyn Error>> {
        let id = match entity {
            "svid" => {
                let id = self.svid_start;
                self.svid_start += 1;
                id.to_string()
            }
            "rptid" => {
                let id = self.rptid_start;
                self.rptid_start += 1;
                id.to_string()
            }
            "ceid" => {
                let id = self.ceid_start;
                self.ceid_start += 1;
                id.to_string()
            }
            "ecid" => {
                let id = self.ecid_start;
                self.ecid_start += 1;
                id.to_string()
            }
            _ => return Err("error entity!!!".into())
        };

        Ok(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_id() {
        let mut id_generator = IDGenerator::new(999);
        println!("id: {:#?}", id_generator);
        // Test generating svid
        for i in 0..5 {
            let id = id_generator.generate_id("svid").unwrap();
            assert_eq!(id, (1000 + i).to_string());
        }

        // Test generating rptid
        for i in 0..5 {
            let id = id_generator.generate_id("rptid").unwrap();
            assert_eq!(id, (2000 + i).to_string());
        }

        // Test generating ceid
        for i in 0..5 {
            let id = id_generator.generate_id("ceid").unwrap();
            assert_eq!(id, (3000 + i).to_string());
        }

        // Test generating ecid
        for i in 0..5 {
            let id = id_generator.generate_id("ecid").unwrap();
            assert_eq!(id, (4000 + i).to_string());
        }

        // Test generating error for unknown entity
        let result = id_generator.generate_id("invalid_entity");
        assert!(result.is_err());
    }
}