/// 使用工厂方法模式设计一个程序来读取各种不同类型的图片格式，
/// 针对每一种图片格式都设计一个图片读取器，如GIF图片读取器用
/// 于读取GIF格式的图片、JPG图片读取器用于读取JPG格式的图片。
/// 需充分考虑系统的灵活性和可扩展性。


// 定义图片读取器 trait
trait ImageReader {
    fn read_image(&self, filename: &str) -> Result<(), String>;
}

// GIF 图片读取器
struct GifImageReader;

impl ImageReader for GifImageReader {
    fn read_image(&self, filename: &str) -> Result<(), String> {
        println!("Reading GIF image: {}", filename);
        // 在这里实现读取 GIF 图片的逻辑
        Ok(())
    }
}

// JPG 图片读取器
struct JpgImageReader;

impl ImageReader for JpgImageReader {
    fn read_image(&self, filename: &str) -> Result<(), String> {
        println!("Reading JPG image: {}", filename);
        // 在这里实现读取 JPG 图片的逻辑
        Ok(())
    }
}

// 定义图片格式枚举
enum ImageFormat {
    Gif,
    Jpg,
    // 添加其他图片格式的枚举项
}

// 图片读取器工厂 trait
trait ImageReaderFactory {
    fn create_image_reader(&self, format: ImageFormat) -> Box<dyn ImageReader>;
}

// 具体的图片读取器工厂实现
struct ConcreteImageReaderFactory;

impl ImageReaderFactory for ConcreteImageReaderFactory {
    fn create_image_reader(&self, format: ImageFormat) -> Box<dyn ImageReader> {
        match format {
            ImageFormat::Gif => Box::new(GifImageReader),
            ImageFormat::Jpg => Box::new(JpgImageReader),
            // 可以根据需要添加其他图片格式的创建逻辑
        }
    }
}

// 客户端代码
fn main() {
    let factory = ConcreteImageReaderFactory;

    // 从文件名获取图片格式（这里简化为字符串）
    let filename = "example.gif";
    let image_format = match filename.split('.').last() {
        Some("gif") => ImageFormat::Gif,
        Some("jpg") | Some("jpeg") => ImageFormat::Jpg,
        _ => {
            println!("Unsupported image format");
            return;
        }
    };

    // 使用工厂创建对应的图片读取器
    let reader = factory.create_image_reader(image_format);

    // 读取图片
    match reader.read_image(filename) {
        Ok(_) => println!("Image successfully read"),
        Err(err) => eprintln!("Failed to read image: {}", err),
    }
}
