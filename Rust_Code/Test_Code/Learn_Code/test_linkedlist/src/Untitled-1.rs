// use std::fmt;

// // 定义链表节点
// struct Node<T> {
//     value: T,
//     next: Option<Box<Node<T>>>,
// }

// // 定义链表
// pub struct LinkedList<T> {
//     head: Option<Box<Node<T>>>,
//     tail: *mut Node<T>, // 使用裸指针指向尾部节点
//     length: usize, // 链表的长度
// }

// impl<T> LinkedList<T> {
//     // 创建一个空链表
//     pub fn new() -> Self {
//         LinkedList {
//             head: None,
//             tail: std::ptr::null_mut(),
//             length: 0, // 初始长度为 0
//         }
//     }

//     // 在链表头部插入一个元素（头插法）
//     pub fn push_front(&mut self, value: T) {
//         let mut new_node = Box::new(Node {
//             value,
//             next: self.head.take(),
//         });

//         // 如果链表为空，更新尾指针
//         if self.tail.is_null() {
//             self.tail = &mut *new_node;
//         }

//         self.head = Some(new_node);
//         self.length += 1; // 长度加一
//     }

//     // 在链表尾部插入一个元素（尾插法）
//     pub fn push_back(&mut self, value: T) {
//         let mut new_node = Box::new(Node { value, next: None });

//         let raw_tail: *mut Node<T> = &mut *new_node;

//         if !self.tail.is_null() {
//             // 如果链表不为空，将新节点链接到尾部
//             unsafe {
//                 (*self.tail).next = Some(new_node);
//             }
//         } else {
//             // 如果链表为空，更新头指针
//             self.head = Some(new_node);
//         }

//         // 更新尾指针
//         self.tail = raw_tail;
//         self.length += 1; // 长度加一
//     }

//     // 从链表头部移除一个元素
//     pub fn pop_front(&mut self) -> Option<T> {
//         self.head.take().map(|node| {
//             self.head = node.next;

//             // 如果链表为空，重置尾指针
//             if self.head.is_none() {
//                 self.tail = std::ptr::null_mut();
//             }

//             self.length -= 1; // 长度减一
//             node.value
//         })
//     }

//     // 删除链表中第一个匹配的元素
//     pub fn remove(&mut self, target: &T) -> bool
//     where
//         T: PartialEq,
//     {
//         let mut current = &mut self.head;

//         // 处理头节点匹配的情况
//         while let Some(ref mut node) = current {
//             if node.value == *target {
//                 // 删除匹配的节点
//                 *current = node.next.take();

//                 // 如果删除的是尾节点，更新尾指针
//                 if current.is_none() {
//                     self.tail = std::ptr::null_mut();
//                 }

//                 self.length -= 1; // 长度减一
//                 return true;
//             }
//             // 移动到下一个节点
//             current = &mut current.as_mut().unwrap().next;
//         }

//         false
//     }

//     // 反转链表
//     pub fn reverse(&mut self) {
//         let mut prev = None;
//         let mut current = self.head.take();

//         while let Some(mut node) = current {
//             let next = node.next.take();
//             node.next = prev.take();
//             prev = Some(node);
//             current = next;
//         }

//         // 更新头指针和尾指针
//         self.head = prev;
//         self.update_tail();
//     }

//     // 更新尾指针
//     fn update_tail(&mut self) {
//         let mut current = &self.head;
//         while let Some(node) = current {
//             if node.next.is_none() {
//                 let node1 = &**node;
//                 self.tail = &**node as *const Node<T> as *mut Node<T>;
//                 break;
//             }
//             current = &node.next;
//         }
//     }

//     // 查看链表头部的元素
//     pub fn peek_front(&self) -> Option<&T> {
//         self.head.as_ref().map(|node| &node.value)
//     }

//     // 查看链表尾部的元素
//     pub fn peek_back(&self) -> Option<&T> {
//         if self.tail.is_null() {
//             None
//         } else {
//             unsafe { Some(&(*self.tail).value) }
//         }
//     }

//     // 查找元素是否存在
//     pub fn contains(&self, target: &T) -> bool
//     where
//         T: PartialEq,
//     {
//         let mut current = &self.head;

//         while let Some(node) = current {
//             if node.value == *target {
//                 return true; // 找到元素
//             }
//             current = &node.next;
//         }

//         false // 没有找到元素
//     }

//     // 获取链表的长度
//     pub fn len(&self) -> usize {
//         self.length
//     }
// }

// // 实现 Display trait 以便打印链表
// impl<T: fmt::Display> fmt::Display for LinkedList<T> {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         let mut current = &self.head;
//         write!(f, "[")?;
//         while let Some(node) = current {
//             write!(f, "{}", node.value)?;
//             current = &node.next;
//             if current.is_some() {
//                 write!(f, ", ")?;
//             }
//         }
//         write!(f, "]")
//     }
// }

// fn main() {
//     let mut list = LinkedList::new();

//     // 测试头插法
//     list.push_front(1);
//     list.push_front(2);
//     list.push_front(3);
//     println!("List after push_front: {}", list); // 输出: List after push_front: [3, 2, 1]

//     // 测试尾插法
//     list.push_back(4);
//     list.push_back(5);
//     println!("List after push_back: {}", list); // 输出: List after push_back: [3, 2, 1, 4, 5]

//     // 测试查找元素
//     println!("Contains 3? {}", list.contains(&3)); // 输出: Contains 3? true
//     println!("Contains 6? {}", list.contains(&6)); // 输出: Contains 6? false

//     // 测试删除元素
//     list.remove(&2);
//     println!("List after removing 2: {}", list); // 输出: List after removing 2: [3, 1, 4, 5]

//     // 测试反转链表
//     list.reverse();
//     println!("List after reverse: {}", list); // 输出: List after reverse: [5, 4, 1, 3]

//     // 测试查看头部和尾部元素
//     println!("Peek front: {:?}", list.peek_front()); // 输出: Peek front: Some(5)
//     println!("Peek back: {:?}", list.peek_back());   // 输出: Peek back: Some(3)
// }