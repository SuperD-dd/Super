/// 使用简单工厂模式设计一个可以创建不同几何形状（如圆形、方形和三角形等）的绘图工具，
/// 每个几何图形都具有绘制draw()和擦除erase()两个方法，要求在绘制不支持的几何图形时，
/// 提示一个UnSupportedShapeException。

// run "cargo run --bin 01"

// 定义几何形状 trait
trait Shape {
    fn draw(&self);
    fn erase(&self);
}

// 定义圆形结构体
struct Circle;

impl Shape for Circle {
    fn draw(&self) {
        println!("Draw a circle");
    }

    fn erase(&self) {
        println!("Erase a circle");
    }
}

// 定义方形结构体
struct Square;

impl Shape for Square {
    fn draw(&self) {
        println!("Draw a square");
    }

    fn erase(&self) {
        println!("Erase a square");
    }
}

// 定义三角形结构体
struct Triangle;

impl Shape for Triangle {
    fn draw(&self) {
        println!("Draw a triangle");
    }

    fn erase(&self) {
        println!("Erase a triangle");
    }
}

// 定义几何图形类型枚举
enum ShapeType {
    Circle,
    Square,
    Triangle,
}

// 简单工厂模式，用于创建不同的几何形状对象
struct ShapeFactory;

impl ShapeFactory {
    fn create_shape(shape_type: ShapeType) -> Result<Box<dyn Shape>, String> {
        match shape_type {
            ShapeType::Circle => Ok(Box::new(Circle)),
            ShapeType::Square => Ok(Box::new(Square)),
            ShapeType::Triangle => Err(String::from("Unsupported shape: Triangle")),
        }
    }
}

// 主函数，演示使用绘图工具绘制不同的几何形状
fn main() {
    let shapes = vec![
        ShapeFactory::create_shape(ShapeType::Circle),
        ShapeFactory::create_shape(ShapeType::Square),
        ShapeFactory::create_shape(ShapeType::Triangle),
    ];

    for shape in shapes {
        match shape {
            Ok(shape) => {
                shape.draw();
                shape.erase();
            }
            Err(err) => {
                eprintln!("{}", err);
            }
        }
    }
}
