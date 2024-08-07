//! `SeaORM` Entity. Generated by sea-orm-codegen 0.12.15

use sea_orm::entity::prelude::*;

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Eq)]
#[sea_orm(table_name = "subject_category")]
pub struct Model {
    #[sea_orm(primary_key)]
    pub id: i64,
    pub category_name: Option<String>,
    pub category_type: Option<i8>,
    pub image_url: Option<String>,
    pub parent_id: Option<i64>,
    pub created_by: Option<String>,
    pub created_time: Option<DateTime>,
    pub update_by: Option<String>,
    pub update_time: Option<DateTime>,
    pub is_deleted: Option<i8>,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

impl ActiveModelBehavior for ActiveModel {}
