use actix_web::{web, App, HttpServer, HttpResponse, Responder};
use log::{error, info, LevelFilter};
use log4rs;
use std::{env, sync::Arc};
use dotenv::dotenv;
use prometheus::{Encoder, TextEncoder, register_int_counter, IntCounter, register_gauge, Gauge};
use tokio::task;
use sys_info;

async fn greet() -> impl Responder {
    info!("Greeting endpoint hit");
    HttpResponse::Ok().body("Hello, Actix Web with log4rs!")
}

async fn metrics(request_counter: web::Data<Arc<IntCounter>>, cpu_usage_gauge: web::Data<Arc<Gauge>>) -> impl Responder {
    let encoder = TextEncoder::new();

    // 将当前 CPU 使用率写入 Gauge
    let cpu_usage = sys_info::loadavg().unwrap().one; // 获取当前 1 分钟平均负载
    info!("CPU usage: {}", cpu_usage);
    cpu_usage_gauge.set(cpu_usage);

    let metric_families = prometheus::gather();
    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer).unwrap();

    HttpResponse::Ok()
        .content_type("text/plain; charset=utf-8")
        .body(buffer)
}

/*从零到一搭建一个rust项目, 支持要求
- 读配置文件(比如说读出来数据库dsn、http端口、日志输出情况)   ok
- 搭建一个http服务，接入Prometheus SDK, 比如说汇报CPU使用率
- 日志支持同时配置化格式，sink(控制台、日志切割、or both)
 */
#[actix_web::main]
async fn main() -> std::io::Result<()> {
    dotenv().ok();
    let database_url = env::var("DATABASE_URL").expect("DATABASE_URL not set in .env");
    let http_port = env::var("HTTP_PORT").unwrap_or_else(|_| "8080".to_string());
    let log4rs_config = env::var("LOG4RS_CONFIG").unwrap_or_else(|_| "log4rs.yaml".to_string());
    let initial_log_level = env::var("INITIAL_LOG_LEVEL").unwrap_or_else(|_| "info".to_string());
    println!("Database URL: {}", database_url);
    let _log_level = match initial_log_level.to_lowercase().as_str() {
        "trace" => LevelFilter::Trace,
        "debug" => LevelFilter::Debug,
        "info" => LevelFilter::Info,
        "warn" => LevelFilter::Warn,
        "error" => LevelFilter::Error,
        _ => LevelFilter::Info, // 默认使用 info 级别
    };

    // 初始化 log4rs 配置
    log4rs::init_file(log4rs_config.clone(), Default::default()).unwrap();
    info!("database_url:{}", database_url);
    info!("http_port:{}", http_port);
    info!("log4rs_config:{}", log4rs_config);
    info!("initial_log_level:{}", initial_log_level);
    info!("_log_level:{}", _log_level);
    // 记录服务器启动日志
    info!("Starting Actix Web server...");

    let request_counter = Arc::new(register_int_counter!("http_requests_total", "Total number of HTTP requests").unwrap());
    let cpu_usage_gauge = Arc::new(register_gauge!("cpu_usage", "Current CPU usage").unwrap());

    // 创建一个线程定时更新 CPU 使用率
    let cpu_usage_gauge_clone = Arc::clone(&cpu_usage_gauge);
    task::spawn(async move {
        loop {
            if let Ok(cpu_usage) = sys_info::loadavg() {
                cpu_usage_gauge_clone.set(cpu_usage.one);
            }
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
        }
    });

    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(Arc::clone(&request_counter)))
            .app_data(web::Data::new(Arc::clone(&cpu_usage_gauge)))
            .route("/", web::get().to(greet))
            .route("/metrics", web::get().to(metrics))
    })
    .bind(format!("127.0.0.1:{}", http_port))?
    .run()
    .await
}
