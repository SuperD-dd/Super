use std::fs::{self, DirEntry};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use regex::Regex;

fn main() -> io::Result<()> {
    let paths = vec![
        // r"D:\flexmi\work\project\src\Foundation.Service.Solution\Foundation.Service.Solution\SystemResource\Link\Serial"
        // r"D:\flexmi\work\project\src\Foundation.Service.Solution\Foundation.Service.Solution\SystemResource\Link_arm64\Serial"
        r"F:\文件\software\fsu7.3\FStudio-Unified\resources\app\lib\backend\SystemResource\Link\Serial"
        ]; // 替换为你的文件夹路径

    // let paths = vec![r"C:\Users\DELL\Desktop\666"]; // 替换为你的文件夹路径

    for path in paths {
        let dir = Path::new(path);
        if dir.is_dir() {
            if let Err(e) = visit_dirs(dir) {
                eprintln!("Error processing directory {:?}: {}", dir, e);
            }
        }
    }
    Ok(())
}

fn visit_dirs(dir: &Path) -> io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            if let Err(e) = visit_dirs(&path) {
                eprintln!("Error processing directory {:?}: {}", path, e);
            }
        } else if path.is_file() {
            if let Err(e) = process_file(&path) {
                eprintln!("Error processing file {:?}: {}", path, e);
            }
        }
    }
    Ok(())
}

fn process_file(path: &Path) -> io::Result<()> {
    let file_name = path.file_name().unwrap().to_str().unwrap();
    let new_file_name = add_suffix_to_file_name(file_name);
    let new_path = path.with_file_name(new_file_name);

    if let Err(e) = fs::rename(path, &new_path) {
        eprintln!("Error renaming file {:?} to {:?}: {}", path, new_path, e);
        return Err(e);
    }

    if new_path.extension().and_then(|s| s.to_str()) == Some("cfg") {
        if let Err(e) = modify_cfg_file(&new_path) {
            eprintln!("Error modifying file {:?}: {}", new_path, e);
            return Err(e);
        }
    }

    Ok(())
}

fn add_suffix_to_file_name(file_name: &str) -> String {
    if let Some(pos) = file_name.rfind('.') {
        format!("{}(NOT SUPPORTED){}", &file_name[..pos], &file_name[pos..])
    } else {
        format!("{}(NOT SUPPORTED)", file_name)
    }
}

fn modify_cfg_file(path: &Path) -> io::Result<()> {
    let mut content = String::new();
    fs::File::open(path)?.read_to_string(&mut content)?;

    let re = Regex::new(r#"(?P<before>PLCName=")(?P<name>[^"]+)(?P<after>")"#).unwrap();
    let modified_content = re.replace_all(&content, r#"${before}${name}(NOT SUPPORTED)${after}"#);

    let mut file = fs::File::create(path)?;
    file.write_all(modified_content.as_bytes())?;
    Ok(())
}

// use std::fs::{self, DirEntry};
// use std::io;
// use std::path::{Path, PathBuf};
// use regex::Regex;

// fn main() -> io::Result<()> {
//     let paths = vec![r"D:\flexmi\work\project\src\Foundation.Service.Solution\Foundation.Service.Solution\SystemResource\Link_windows"]; 

//     for path in paths {
//         let dir = Path::new(path);
//         if dir.is_dir() {
//             if let Err(e) = visit_dirs(dir) {
//                 eprintln!("Error processing directory {:?}: {}", dir, e);
//             }
//         }
//     }
//     Ok(())
// }

// fn visit_dirs(dir: &Path) -> io::Result<()> {
//     for entry in fs::read_dir(dir)? {
//         let entry = entry?;
//         let path = entry.path();
//         if path.is_dir() {
//             if let Err(e) = visit_dirs(&path) {
//                 eprintln!("Error processing directory {:?}: {}", path, e);
//             }
//         } else if path.is_file() {
//             if let Err(e) = process_file(&path) {
//                 eprintln!("Error processing file {:?}: {}", path, e);
//             }
//         }
//     }
//     Ok(())
// }

// fn process_file(path: &Path) -> io::Result<()> {
//     let file_name = path.file_name().unwrap().to_str().unwrap();

//     if file_name.ends_with("(NOT SUPPORTED).so") || file_name.ends_with("(NOT SUPPORTED).dll") {
//         if let Err(e) = fs::remove_file(path) {
//             eprintln!("Error deleting file {:?}: {}", path, e);
//         } else {
//             println!("Deleted file: {:?}", path);
//         }
//     }
    
//     Ok(())
// }

