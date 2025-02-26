#[derive(Debug, Clone)]
struct Node<T> {
    value: T,
    next: Option<Box<Node<T>>>,
}

#[derive(Debug, Clone)]
struct LinkedList<T> {
    head: Option<Box<Node<T>>>,
    length: usize,
}

impl<T> LinkedList<T> {
    fn new() -> Self {
        LinkedList {
            head: None,
            length: 0,
        }
    }

    fn new_from_vec(vec: Vec<T>) -> Self {
        let mut linked_list = LinkedList::new();
        for value in vec.into_iter().rev() {
            linked_list.push(value);
        }
        linked_list
    }

    fn push(&mut self, value: T) {
        let mut new_node = Box::new(Node {
            value,
            next: None,
        });

        match self.head.take() {
            Some(node) => {
                new_node.next = Some(node);
                self.head = Some(new_node);
            }
            None => {
                self.head = Some(new_node);
            }
        }
        self.length += 1;
    }

    fn reverse(&mut self) {
        let mut prev = None;
        let mut curr = self.head.take();
        while let Some(mut node) = curr {
            let next = node.next.take();
            node.next = prev;
            prev = Some(node);
            curr = next;
        }
        self.head = prev;
    }

    // 从链表头部移除一个元素
    pub fn pop_front(&mut self) -> Option<T> {
        self.head.take().map(|node| {
            self.head = node.next;
            self.length -= 1; // 长度减一
            node.value
        })
    }

    // 增加 Display 约束，使得可以打印 LinkedList
    fn print(&self)
    where
        T: std::fmt::Display, // 只有实现了 Display 的类型才能打印
    {
        let mut current = &self.head;
        while let Some(node) = current {
            print!("{} -> ", node.value);
            current = &node.next;
        }
        println!("None");
    }
}

// Function to add two linked lists (right-aligned)
// fn linked_list_add(l1: &mut LinkedList<i32>, l2: &mut LinkedList<i32>) -> LinkedList<i32> {
//     l1.reverse();
//     l2.reverse();
//     let mut result = LinkedList::new();
//     let mut carry = 0;
//     let mut ll1 = l1.pop_front();  // Use pop_front to access and mutate the first element
//     let mut ll2 = l2.pop_front();
//     while ll1.is_some() || ll2.is_some() || carry != 0 {
//         let val1 = ll1.take().unwrap_or(0);  // Default to 0 if no more nodes
//         let val2 = ll2.take().unwrap_or(0);  // Default to 0 if no more nodes

//         let sum = val1 + val2 + carry;
//         carry = sum / 10;  // Carry over
//         result.push(sum % 10);  // Add current digit to result

//         // Move to the next nodes, if available
//         ll1 = l1.pop_front();
//         ll2 = l2.pop_front();
//     }

//     result

// }

// 假设LinkedList已实现Clone和Reverse方法

fn linked_list_add(l1: &LinkedList<i32>, l2: &LinkedList<i32>) -> LinkedList<i32> {
    // 克隆链表以避免修改原始数据
    let mut l1_clone = l1.clone();
    let mut l2_clone = l2.clone();
    
    // 反转链表以便从低位开始相加
    l1_clone.reverse();
    l2_clone.reverse();
    
    let mut result = LinkedList::new();
    let mut carry = 0;
    
    let mut current1 = l1_clone.head.as_ref();
    let mut current2 = l2_clone.head.as_ref();
    
    // 遍历所有节点，直到两个链表都处理完毕且无进位
    while current1.is_some() || current2.is_some() || carry > 0 {
        let mut sum = carry;
        
        // 处理l1的当前节点
        if let Some(node) = current1 {
            sum += node.value;
            current1 = node.next.as_ref();
        }
        
        // 处理l2的当前节点
        if let Some(node) = current2 {
            sum += node.value;
            current2 = node.next.as_ref();
        }
        
        // 计算进位和当前位的值
        carry = sum / 10;
        result.push(sum % 10);
    }
    
    // 反转结果链表，恢复高位在前
    result
}

fn main() {
    let vec1 = vec![1, 2, 4, 8];  // Represents the number 8241
    let vec2 = vec![2, 3, 5];     // Represents the number 532

    let mut l1 = LinkedList::new_from_vec(vec1);
    let mut l2 = LinkedList::new_from_vec(vec2);

    // 打印链表
    println!("l1 = ");
    l1.print();
    println!("l2 = ");
    l2.print();

    let result = linked_list_add(&mut l1.clone(), &mut l2.clone());

    println!("Sum = ");
    result.print();
    // 打印链表
    println!("l1 = ");
    l1.print();
    println!("l2 = ");
    l2.print();
}


// use std::ops::Deref;

// type NodePtr<T> = Option<Box<Node<T>>>;
// struct Node<T> {
//     data: T,
//     next: NodePtr<T>,
// }

// impl<T> Node<T> {
//     fn new(data: T, next: NodePtr<T>) -> Self {
//         Self { data, next }
//     }
//     fn from_vec(vec: Vec<T>) -> NodePtr<T> {
//         let mut head = None;
//         for elem in vec.into_iter().rev() {
//             let node = Box::new(Node::new(elem, head.take()));
//             head.replace(node);
//         }
//         head
//     }
// }

// fn into_vec<T>(mut head: NodePtr<T>) -> Vec<T> {
//     let mut result = Vec::new();
//     while let Some(node) = head.take() {
//         result.push(node.data);
//         head = node.next;
//     }
//     result
// }

// fn reverse<T>(mut head: Box<Node<T>>, mut f: impl FnMut(&mut T)) -> Box<Node<T>> {
//     f(&mut head.data);
//     let mut next = head.next.take();
//     while let Some(mut node) = next {
//         next = node.next.replace(head);
//         head = node;
//         f(&mut head.data);
//     }
//     head
// }

// fn add(a: NodePtr<u32>, b: NodePtr<u32>) -> NodePtr<u32> {
//     match (a, b) {
//         (Some(a), Some(b)) => {
//             let (mut a_len, mut b_len) = (0, 0);
//             let a = reverse(a, |_| a_len += 1);
//             let b = reverse(b, |_| b_len += 1);
//             let (a, b) = if a_len > b_len {
//                 (a, b)
//             } else {
//                 (b, a)
//             };
//             let mut b_ref = Some(b.deref());
//             let mut carry = 0;
//             let mut head = reverse(a, |digit| {
//                 if let Some(b) = b_ref {
//                     *digit += b.data;
//                     b_ref = b.next.as_deref();
//                 }
//                 *digit += carry;
//                 carry = if *digit >= 10 {
//                     *digit -= 10;
//                     1
//                 } else {
//                     0
//                 };
//             });
//             if carry != 0 {
//                 head = Box::new(Node::new(1, Some(head)));
//             }
//             Some(head)
//         }
//         (Some(a), None) | (None, Some(a)) => Some(a),
//         (None, None) => None,
//     }
// }

// #[test]
// fn test_list() {
//     fn test_add(a: Vec<u32>, b: Vec<u32>, expected: Vec<u32>) {
//         let a = Node::from_vec(a);
//         let b = Node::from_vec(b);
//         assert_eq!(into_vec(add(a, b)), expected);
//     }
//     test_add(vec![1, 2, 3, 4], vec![2, 3, 4], vec![1, 4, 6, 8]);
//     test_add(vec![], vec![2, 3, 4], vec![2, 3, 4]);
//     test_add(vec![1, 2, 3, 4], vec![], vec![1, 2, 3, 4]);
//     test_add(vec![1], vec![], vec![1]);
//     test_add(vec![9], vec![1], vec![1, 0]);
//     test_add(vec![9, 9, 9, 9], vec![1], vec![1, 0, 0, 0, 0]);
// }
