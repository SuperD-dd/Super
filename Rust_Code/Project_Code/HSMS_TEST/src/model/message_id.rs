pub struct MessageID {
    index: i32,
}

impl MessageID {
    pub fn new() -> Self {
        MessageID { index: 0 }
    }

    pub fn get_id(&mut self) -> i32 {
        self.index += 1;
        if self.index > 254 {
            self.index = 0;
        }
        self.index
    }
}


