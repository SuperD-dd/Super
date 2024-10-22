use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Default, Clone)]
#[serde(rename = "Secs")]
pub struct XMLSecs {
    #[serde(rename = "SVID")]
    pub svids: XMLSVIDs,
    #[serde(rename = "ECID")]
    pub ecids: XMLECIDs,
    #[serde(rename = "RPTID")]
    pub rptids: XMLRPTIDs,
    #[serde(rename = "CEID")]
    pub ceids: XMLCEIDs,
}

#[derive(Serialize, Deserialize, Debug, Default, Clone)]
pub struct XMLSVIDs {
    #[serde(rename = "SVID")]
    pub svid: Vec<XMLSVID>,
}

#[derive(Serialize, Deserialize, Debug, Default, Clone)]
pub struct XMLSVID {
    #[serde(rename = "ID")]
    pub id: String,
    #[serde(rename = "SVName")]
    pub sv_name: String,
    #[serde(rename = "VALUES")]
    pub values: String,
    #[serde(rename = "UNITS")]
    pub units: String,
}

#[derive(Serialize, Deserialize, Debug, Default, Clone)]
pub struct XMLECIDs {
    #[serde(rename = "ECID")]
    pub ecid: Vec<XMLECID>,
}

#[derive(Serialize, Deserialize, Debug, Default, Clone)]
pub struct XMLECID {
    #[serde(rename = "ID")]
    pub id: String,
    #[serde(rename = "Name")]
    pub name: String,
    #[serde(rename = "Value")]
    pub value: String,
}

#[derive(Serialize, Deserialize, Debug, Default, Clone)]
pub struct XMLRPTIDs {
    #[serde(rename = "RPTID")]
    pub rptid: Vec<XMLRPTID>,
}

#[derive(Serialize, Deserialize, Debug, Default, Clone)]
pub struct XMLRPTID {
    #[serde(rename = "ID")]
    pub id: String,
    #[serde(rename = "Name")]
    pub name: String,
    #[serde(rename = "SVID")]
    pub vid: Vec<XMLSVIDRef>,
}

#[derive(Serialize, Deserialize, Debug, Default, Clone)]
pub struct XMLSVIDRef {
    #[serde(rename = "ID")]
    pub id: String,
}

#[derive(Serialize, Deserialize, Debug, Default, Clone)]
pub struct XMLCEIDs {
    #[serde(rename = "CEID")]
    pub ceid: Vec<XMLCEID>,
}

#[derive(Serialize, Deserialize, Debug, Default, Clone)]
pub struct XMLCEID {
    #[serde(rename = "ID")]
    pub id: String,
    #[serde(rename = "Name")]
    pub name: String,
    #[serde(rename = "Enable")]
    pub enable: bool,
    #[serde(rename = "RPTID")]
    pub rptid: Option<Vec<XMLRPTIDRef>>,
}

#[derive(Serialize, Deserialize, Debug, Default, Clone)]
pub struct XMLRPTIDRef {
    #[serde(rename = "ID")]
    pub id: String,
}