use actix_web::{middleware, web, App, HttpServer};
pub mod auth;
#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(move || {
        App::new()
            .wrap(middleware::Logger::default())
            .wrap(auth::Auth)
            // 注册其他路由和处理函数
            .route("/", web::get().to(|| async { "Hello, World!" }))
            .route("/login", web::get().to(|| async { "Hello, login" }))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}