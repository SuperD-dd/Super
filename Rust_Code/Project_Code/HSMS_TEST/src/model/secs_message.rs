use crate::model::secs_body::SecsBody;
use crate::model::secs_header::SecsHeader;
#[derive(Debug)]
pub struct SecsMessage {
    length: i32,
    is_request: bool,   //
    header: SecsHeader,
    body: SecsBody,
    //t3: TimeT3, // Assuming TimeT3 is a custom type
}

impl SecsMessage {
    pub fn new(header: SecsHeader, body: SecsBody, is_request: bool) -> Self {
        SecsMessage {
            length: 0,
            is_request,
            header,
            body,
            //t3: TimeT3::new(), // Assuming TimeT3 has a new() constructor
        }
    }

    pub fn set_is_request(&mut self, value: bool) {
        self.is_request = value;
    }

    pub fn set_length(&mut self, value: i32) {
        self.length = value;
    }

    pub fn header(&self) -> &SecsHeader {
        &self.header
    }

    pub fn body(&self) -> &SecsBody {
        &self.body
    }

    /*pub fn t3(&self) -> &TimeT3 {
        &self.t3
    }

    pub fn t3_mut(&mut self) -> &mut TimeT3 {
        &mut self.t3
    }*/
}

// Assuming you have defined TimeT3 and other structs elsewhere in your code
