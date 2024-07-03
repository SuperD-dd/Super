mod entities;
use entities::{prelude::*, *};
use sea_orm::*;
use futures::executor::block_on;


const DATABASE_URL: &str = "mysql://root:Wing1Q2W%23E@121.43.163.210:3306";
const DB_NAME: &str = "dd-club";

async fn run() -> Result<(), DbErr> {
    let db = Database::connect(DATABASE_URL).await?;

    let db = &match db.get_database_backend() {
                DbBackend::MySql => {
                   db.execute(Statement::from_string(
                       db.get_database_backend(),
                       format!("CREATE DATABASE IF NOT EXISTS `{}`;", DB_NAME),
                   ))
               .await?;
        
                   let url = format!("{}/{}", DATABASE_URL, DB_NAME);
                   Database::connect(&url).await?
              }
                DbBackend::Postgres => {
                   db.execute(Statement::from_string(
                       db.get_database_backend(),
                       format!("DROP DATABASE IF EXISTS \"{}\";", DB_NAME),
                   ))
                   .await?;
                   db.execute(Statement::from_string(
                       db.get_database_backend(),
                       format!("CREATE DATABASE \"{}\";", DB_NAME),
                   ))
                   .await?;
        
                   let url = format!("{}/{}", DATABASE_URL, DB_NAME);
                   Database::connect(&url).await?
                }
                DbBackend::Sqlite => db,
           };

    let happy_bakery = auth_permission::ActiveModel {
        id: ActiveValue::Set(5),
        name: ActiveValue::set(Some("ddd222".to_string()).to_owned()),
        ..Default::default()
    };
    happy_bakery.update(db).await?;
    let bakeries: Vec<auth_permission::Model> = AuthPermission::find().all(db).await?;
    println!("bakeries: {:#?}", bakeries);
    Ok(())
}

fn main() {
    if let Err(err) = block_on(run()) {
        panic!("{}", err);
    }
}
