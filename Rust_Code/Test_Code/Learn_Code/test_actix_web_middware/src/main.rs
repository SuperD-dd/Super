// use actix_web::{middleware, web, App, HttpServer};
// pub mod auth;
// #[actix_web::main]
// async fn main() -> std::io::Result<()> {
//     HttpServer::new(move || {
//         App::new()
//             .wrap(middleware::Logger::default())
//             .wrap(auth::Auth)
//             // 注册其他路由和处理函数
//             .route("/", web::get().to(|| async { "Hello, World!" }))
//             .route("/login", web::get().to(|| async { "Hello, login" }))
//     })
//     .bind("127.0.0.1:8080")?
//     .run()
//     .await
// }

/// path_params
// use actix_web::{get, App, HttpRequest, HttpServer, Result};

// #[get("/a/{v1}/{v2}/")]
// async fn index(req: HttpRequest) -> Result<String> {
//     println!("{:?}", req);
//     let v1: u8 = req.match_info().get("v1").unwrap().parse().unwrap();
//     let v2: u8 = req.match_info().query("v2").parse().unwrap();
//     let (v3, v4): (u8, u8) = req.match_info().load().unwrap();
//     Ok(format!("Values {} {} {} {}", v1, v2, v3, v4))
// }

// #[actix_web::main]
// async fn main() -> std::io::Result<()> {
//     HttpServer::new(|| App::new().service(index))
//         .bind(("127.0.0.1", 8080))?
//         .run()
//         .await
// }

/// path_info
// use actix_web::{get, web, App, HttpServer, Result};

// #[get("/{username}/{id}/index.html")] // <- define path parameters
// async fn index(info: web::Path<(String, u32)>) -> Result<String> {
//     println!("{:?}", info);
//     let info = info.into_inner();
//     Ok(format!("Welcome {}! id: {}", info.0, info.1))
// }

// #[actix_web::main]
// async fn main() -> std::io::Result<()> {
//     HttpServer::new(|| App::new().service(index))
//         .bind(("127.0.0.1", 8080))?
//         .run()
//         .await
// }

/// url_for
// use actix_web::{get, guard, http::header, HttpRequest, HttpResponse, Result};

// #[get("/test/")]
// async fn index(req: HttpRequest) -> Result<HttpResponse> {
//     println!("{:?}", req);
//     let url = req.url_for("foo", ["1", "2", "3"])?; // <- generate url for "foo" resource
//     println!("{:?}", url);
//     Ok(HttpResponse::Found()
//         .insert_header((header::LOCATION, url.as_str()))
//         .finish())
// }

// #[actix_web::main]
// async fn main() -> std::io::Result<()> {
//     use actix_web::{web, App, HttpServer};

//     HttpServer::new(|| {
//         App::new()
//             .service(
//                 web::resource("/test/{a}/{b}/{c}")
//                     .name("foo") // <- set resource name, then it could be used in `url_for`
//                     .guard(guard::Get())
//                     .to(HttpResponse::Ok),
//             )
//             .service(index)
//     })
//     .bind(("127.0.0.1", 8080))?
//     .run()
//     .await
// }

///外部资源
// use actix_web::{get, App, HttpRequest, HttpServer, Responder};

// #[get("/")]
// async fn index(req: HttpRequest) -> impl Responder {
//     let url = req.url_for("youtube", ["oHg5SJYRHA0"]).unwrap();
//     assert_eq!(url.as_str(), "https://youtube.com/watch/oHg5SJYRHA0");

//     url.to_string()
// }

// #[actix_web::main]
// async fn main() -> std::io::Result<()> {
//     HttpServer::new(|| {
//         App::new()
//             .service(index)
//             .external_resource("youtube", "https://youtube.com/watch/{video_id}")
//     })
//     .bind(("127.0.0.1", 8080))?
//     .run()
//     .await
// }

/// 中间件
// use actix_web::{middleware, HttpResponse};

// async fn index() -> HttpResponse {
//     HttpResponse::Ok().body("Hello")
// }

// #[actix_web::main]
// async fn main() -> std::io::Result<()> {
//     use actix_web::{web, App, HttpServer};

//     HttpServer::new(|| {
//         App::new()
//             .wrap(middleware::NormalizePath::default())
//             .route("/resource/", web::to(index))
//     })
//     .bind(("127.0.0.1", 8080))?
//     .run()
//     .await
// }

/// guard
// use actix_web::{
//     guard::{Guard, GuardContext},
//     http, HttpResponse,
// };

// struct ContentTypeHeader;

// impl Guard for ContentTypeHeader {
//     fn check(&self, req: &GuardContext) -> bool {
//         req.head()
//             .headers()
//             .contains_key(http::header::CONTENT_TYPE)
//     }
// }

// #[actix_web::main]
// async fn main() -> std::io::Result<()> {
//     use actix_web::{web, App, HttpServer};

//     HttpServer::new(|| {
//         App::new().route(
//             "/",
//             web::route().guard(ContentTypeHeader).to(HttpResponse::Ok),
//         )
//     })
//     .bind(("127.0.0.1", 8080))?
//     .run()
//     .await
// }

/// guard
// use actix_web::{guard, web, App, HttpResponse, HttpServer};

// #[actix_web::main]
// async fn main() -> std::io::Result<()> {
//     HttpServer::new(|| {
//         App::new().route(
//             "/",
//             web::route()
//                 .guard(guard::Not(guard::Get()))
//                 .to(HttpResponse::MethodNotAllowed),
//         )
//     })
//     .bind(("127.0.0.1", 8080))?
//     .run()
//     .await
// }

/// extract `Info` using serde
// use actix_web::{web, App, HttpServer, Result};
// use serde::Deserialize;

// #[derive(Deserialize)]
// struct Info {
//     username: String,
// }

// /// extract `Info` using serde
// async fn index(info: web::Json<Info>) -> Result<String> {
//     Ok(format!("Welcome {}!", info.username))
// }

// #[actix_web::main]
// async fn main() -> std::io::Result<()> {
//     HttpServer::new(|| App::new().route("/", web::post().to(index)))
//         .bind(("127.0.0.1", 8080))?
//         .run()
//         .await
// }

/// Content encoding
// use actix_web::{
//     get, http::header::ContentEncoding, middleware, App, HttpResponse, HttpServer,
// };

// #[get("/")]
// async fn index() -> HttpResponse {
//     HttpResponse::Ok()
//         // v- disable compression
//         .insert_header(ContentEncoding::Identity)
//         .body("data")
// }

// #[actix_web::main]
// async fn main() -> std::io::Result<()> {
//     HttpServer::new(|| {
//         App::new()
//             .wrap(middleware::Compress::default())
//             .service(index)
//     })
//     .bind(("127.0.0.1", 8080))?
//     .run()
//     .await
// }

/// session
use actix_session::{Session, SessionMiddleware, storage::CookieSessionStore};
use actix_web::{cookie::Key, web, App, Error, HttpRequest, HttpResponse, HttpServer};

async fn index(req: HttpRequest,session: Session) -> Result<HttpResponse, Error> {
    println!("req: {:?}", req);
    // access session data
    if let Some(cookie) = req.cookie("") {
        println!("Cookie Header: {:?}", cookie);
    }
    if let Some(count) = session.get::<i32>("counter")? {
        session.insert("counter", count + 1)?;
    } else {
        session.insert("counter", 1)?;
    }

    Ok(HttpResponse::Ok().body(format!(
        "Count is {:?}!",
        session.get::<i32>("counter")?.unwrap()
    )))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| {
        App::new()
            .wrap(
                // create cookie based session middleware
                SessionMiddleware::builder(CookieSessionStore::default(), Key::from(&[0; 64]))
                    .cookie_secure(false)
                    .build()
            )
            .service(web::resource("/").to(index))
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}