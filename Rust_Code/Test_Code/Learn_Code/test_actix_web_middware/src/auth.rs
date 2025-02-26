// 添加以下导入
use actix_web::dev::{Service, ServiceRequest, ServiceResponse, Transform, forward_ready};
use actix_web::error;
use actix_web::http::header::HeaderValue;
use std::future::{ready, Ready};
use futures_util::future::LocalBoxFuture;

//中间的工厂类
pub struct Auth;
 
 
impl<S, B> Transform<S, ServiceRequest> for Auth
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = actix_web::Error>,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = actix_web::Error;
    type InitError = ();
    type Transform = AuthMiddleware<S>;
    type Future = Ready<Result<Self::Transform, Self::InitError>>;
    fn new_transform(&self, service: S) -> Self::Future {
        ready(Ok(AuthMiddleware { service }))
    }
}
  
// 中间件的具体实现，里面需要接受工厂类里面过来的service
pub struct AuthMiddleware<S> {
    service: S,
}
 
//具体实现
//核心是两个方法：
// call 具体实现
// poll_ready
impl<S, B> Service<ServiceRequest> for AuthMiddleware<S>
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = actix_web::Error>,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = actix_web::Error;
    type Future = LocalBoxFuture<'static, Result<Self::Response, Self::Error>>;
 
    // 实现 poll_ready 方法，用于检查服务是否准备好处理请求 
    //这里用的是forward_ready!宏
    forward_ready!(service);
 
    // 实现 call 方法，用于处理实际的请求
    fn call(&self, req: ServiceRequest) -> Self::Future {
        // 进行鉴权操作，判断是否有权限
        if has_permission(&req) {
            // 有权限，继续执行后续中间件
            let fut = self.service.call(req);
            Box::pin(async move {
                let res = fut.await?;
                Ok(res)
            })
        } else {
            // 没有权限，立即返回响应
            Box::pin(async move {
                // 鉴权失败，返回未授权的响应，停止后续中间件的调用
                Err(error::ErrorUnauthorized("Unauthorized"))
            })
        }
    }
}
 
fn has_permission(req: &ServiceRequest) -> bool {
    // 实现你的鉴权逻辑，根据需求判断是否有权限
    // 返回 true 表示有权限，返回 false 表示没有权限
    // unimplemented!()
    let value = HeaderValue::from_str("").unwrap();
    match req.path().to_ascii_lowercase().as_str(){
        "/login" => true,
        _ => {
            let token = req.headers().get("token").unwrap_or(&value);
            if token.len() <=0{
                false
            }else{
                println!("验证一下token，看看是否合法");
                true
            }
        }
    }
}