#![allow(dead_code)]
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}
#[warn(unused_mut)]
impl UnionFind {
    fn new(n: usize) -> UnionFind {
        let mut parent = vec![0; n];
        let mut rank = vec![1; n];

        for i in 0..n {
            parent[i] = i;
        }

        UnionFind { parent, rank }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }

        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize) {
        let root_x = self.find(x);
        let root_y = self.find(y);

        if root_x != root_y {
            if self.rank[root_x] < self.rank[root_y] {
                self.parent[root_x] = root_y;
            } else if self.rank[root_x] > self.rank[root_y] {
                self.parent[root_y] = root_x;
            } else {
                self.parent[root_y] = root_x;
                self.rank[root_x] += 1;
            }
        }
    }
}

struct Node<T> {
    value: T,
    next: Option<Box<Node<T>>>,
}

pub struct LinkedList<T> {
    head: Option<Box<Node<T>>>,
    tail: *mut Node<T>,
    length: usize,
}

impl<T> LinkedList<T> {
    pub fn new() -> Self {
        LinkedList {
            head: None,
            tail: std::ptr::null_mut(),
            length: 0,
        }
    }

    pub fn push_front(&mut self, value: T) {
        let mut new_node = Box::new(Node {
            value,
            next: self.head.take(),
        });

        if self.tail.is_null() {
            self.tail = &mut *new_node as *mut Node<T>;
        }

        self.head = Some(new_node);
        self.length += 1;
    }

    pub fn push_back(&mut self, value: T) {
        let mut new_node = Box::new(Node { value, next: None });

        let raw_tail: *mut Node<T> = &mut *new_node;

        if !self.tail.is_null() {
            unsafe {
                (*self.tail).next = Some(new_node);
            }
        } else {
            self.head = Some(new_node);
        }

        self.tail = raw_tail;
        self.length += 1;
    }

    pub fn pop_front(&mut self) -> Option<T> {
        self.head.take().map(|node| {
            self.head = node.next;

            if self.head.is_none() {
                self.tail = std::ptr::null_mut();
            }

            self.length -= 1;
            node.value
        })
    }

    pub fn remove(&mut self, target: &T) -> bool
    where
        T: PartialEq,
    {
        let mut current = &mut self.head;

        while let Some(ref mut node) = current {
            if node.value == *target {
                *current = node.next.take();

                if current.is_none() {
                    self.tail = std::ptr::null_mut();
                }

                self.length -= 1;
                return true;
            }
            current = &mut current.as_mut().unwrap().next;
        }

        false
    }

    pub fn reverse(&mut self) {
        let mut prev = None;
        let mut current = self.head.take();

        while let Some(mut node) = current {
            let next = node.next.take();
            node.next = prev.take();
            prev = Some(node);
            current = next;
        }

        self.head = prev;
        self.update_tail();
    }

    fn update_tail(&mut self) {
        let mut current = &self.head;
        while let Some(node) = current {
            if node.next.is_none() {
                self.tail = &**node as *const Node<T> as *mut Node<T>;
                break;
            }
            current = &node.next;
        }
    }

    pub fn peek_front(&self) -> Option<&T> {
        self.head.as_ref().map(|node| &node.value)
    }

    pub fn peek_back(&self) -> Option<&T> {
        if self.tail.is_null() {
            None
        } else {
            unsafe { Some(&(*self.tail).value) }
        }
    }

    pub fn contains(&self, target: &T) -> bool
    where
        T: PartialEq,
    {
        let mut current = &self.head;

        while let Some(node) = current {
            if node.value == *target {
                return true;
            }
            current = &node.next;
        }

        false
    }

    pub fn len(&self) -> usize {
        self.length
    }
}

// 实现 Display trait 以便打印链表
impl<T: fmt::Display> fmt::Display for LinkedList<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut current = &self.head;
        write!(f, "[")?;
        while let Some(node) = current {
            write!(f, "{}", node.value)?;
            current = &node.next;
            if current.is_some() {
                write!(f, ", ")?;
            }
        }
        write!(f, "]")
    }
}

// 实现一个内存分配器2502
struct Allocator {
    n: usize,
    memory: Vec<i32>,
}

impl Allocator {
    fn new(n: i32) -> Self {
        Allocator {
            n: n as usize,
            memory: vec![0; n as usize],
        }
    }
    
    fn allocate(&mut self, size: i32, m_id: i32) -> i32 {
        let mut count = 0;
        for i in 0..self.n {
            if self.memory[i] != 0 {
                count = 0;
            } else {
                count += 1;
                if count == size {
                    for j in (i as i32 - count + 1)..=i as i32 {
                        self.memory[j as usize] = m_id;
                    }
                    return i as i32 - count + 1;
                }
            }
        }
        -1
    }
    
    fn free_memory(&mut self, m_id: i32) -> i32 {
        let mut count = 0;
        for i in 0..self.n {
            if self.memory[i] == m_id {
                count += 1;
                self.memory[i] = 0;
            }
        }
        count
    }
}

//1472 浏览器浏览记录
struct BrowserHistory {
    history: Vec<String>,
    current_index: usize,
}

/** 
 * `&self` means the method takes an immutable reference.
 * If you need a mutable reference, change it to `&mut self` instead.
 */
impl BrowserHistory {

    fn new(homepage: String) -> Self {
        Self{
            history: vec![homepage],
            current_index: 0,
        }
    }
    
    fn visit(&mut self, url: String) {
        if (self.current_index + 1) == self.history.len() {
            self.history.push(url);
        } else {
            self.history[self.current_index + 1] = url;
            self.history.drain((self.current_index + 2)..);
        }
        self.current_index += 1;
    }
    
    fn back(&mut self, steps: i32) -> String {
        if steps >= self.current_index as i32 {
            self.current_index = 0;
            return self.history[0].clone();
        } else {
            self.current_index -= steps as usize;
            return self.history[self.current_index].clone();
        }
    }
    
    fn forward(&mut self, steps: i32) -> String {
        if steps + self.current_index as i32 >= self.history.len() as i32 {
            self.current_index = self.history.len() - 1;
            return self.history[self.current_index].clone();
        } else {
            self.current_index += steps as usize;
            return self.history[self.current_index].clone();
        }
    }
}

use core::{hash, num};
/**
 * Your BrowserHistory object will be instantiated and called as such:
 * let obj = BrowserHistory::new(homepage);
 * obj.visit(url);
 * let ret_2: String = obj.back(steps);
 * let ret_3: String = obj.forward(steps);
 */
use std::cmp::{max, min};

struct BrowserHistory1 {
    urls: Vec<String>,
    curr_index: usize,
}

impl BrowserHistory1 {
    fn new(homepage: String) -> Self {
        BrowserHistory1 {
            urls: vec![homepage],
            curr_index: 0,
        }
    }
    
    fn visit(&mut self, url: String) {
        self.urls.truncate(self.curr_index + 1);
        self.urls.push(url);
        self.curr_index += 1;
    }
    
    fn back(&mut self, steps: i32) -> String {
        self.curr_index = max(self.curr_index as i32 - steps, 0) as usize;
        return self.urls[self.curr_index].clone();
    }
    
    fn forward(&mut self, steps: i32) -> String {
        self.curr_index = std::cmp::min(self.curr_index + steps as usize, self.urls.len() - 1);
        return self.urls[self.curr_index].clone();
    }
}

struct Solution;
//125
impl Solution {
    pub fn is_palindrome(s: String) -> bool {
        let clean_str: String = s.chars().filter(|x| x.is_ascii_alphanumeric()).collect();
        let rever_str: String = clean_str.chars().rev().collect();
        println!("clean_str: {}", clean_str);
        println!("revert_str: {}", rever_str.clone());
        clean_str.eq_ignore_ascii_case(&rever_str)
        // let mut iter = s
        // .chars()
        // .filter(|c| c.is_alphanumeric())
        // .map(|c| c.to_ascii_lowercase());
        // println!("iter: {:?}", iter.clone());
        // iter.clone().eq(iter.rev())
    }
}

//136
impl Solution {
    pub fn single_number(nums: Vec<i32>) -> i32 {
        let mut result = 0;
        for num in nums {
            result ^= num;
        }
        result
    }
}

//2789
impl Solution {
    pub fn max_array_value(nums: Vec<i32>) -> i64 {
        nums.into_iter()
            .map(|x| x as i64)
            .rfold(0, |acc, x| if x <= acc { acc + x } else { x })
    }
}

//2824
impl Solution {
    pub fn count_pairs(nums: Vec<i32>, target: i32) -> i32 {
        let mut ans = 0;
        let n = nums.len();
        for i in 0..n - 1 {
            let a = nums[i];
            for j in i + 1..n {
                if a + nums[j] < target {
                    ans += 1;
                }
            }
        }
        ans
    }
}

//168
impl Solution {
    pub fn convert_to_title(n: i32) -> String {
        let mut result = String::new();
        let mut n = n;

        while n > 0 {
            n -= 1;
            let remainder = n % 26;
            let character = (remainder as u8 + b'A') as char;
            result.push(character);
            n /= 26;
        }

        result.chars().rev().collect()
    }
}

//169
impl Solution {
    pub fn majority_element_1(nums: Vec<i32>) -> i32 {
        use std::{arch::x86_64, collections::HashMap};
        let mut map: HashMap<i32, i32> = HashMap::new();
        let half_len: i32 = (nums.len() / 2) as i32;
        let result = nums.iter().fold(0, |acc, &x| {
            let count = map.entry(x).or_insert(0);
            *count += 1;
            if *count > half_len {
                x
            } else {
                acc
            }
        });
        result
    }
}

//169
impl Solution {
    pub fn majority_element_2(nums: Vec<i32>) -> i32 {
        let mut candidate = 0;
        let mut count = 0;
        for num in nums {
            if count == 0 {
                candidate = num;
                count = 1;
            } else if candidate == num {
                count += 1;
            } else {
                count -= 1;
            }
        }
        candidate
    }
}

//907
impl Solution {
    pub fn sum_subarray_mins(arr: Vec<i32>) -> i32 {
        let mut result = 0;
        let mut stack: Vec<(i32, i32)> = Vec::new();
        let mut prev = 0;
        const MOD: i32 = 1_000_000_007;
        for i in 0..arr.len() {
            let mut count = 1;
            while !stack.is_empty() && stack.last().unwrap().0 >= arr[i] {
                let (val, c) = stack.pop().unwrap();
                count += c;
                prev -= val * c;
            }
            stack.push((arr[i], count));
            prev += arr[i] * count;
            result += prev % MOD;
        }
        result
    }
}

//1631
impl Solution {
    pub fn minimum_effort_path_1(heights: Vec<Vec<i32>>) -> i32 {
        let mut visited = vec![vec![false; heights[0].len()]; heights.len()];
        let mut result = 0;
        let mut left = 0;
        let mut right = 1000000; // 由于高度最大为10^6，所以最大可能的答案为1000000

        while left < right {
            let mid = left + (right - left) / 2;

            if Self::dfs_1631(&heights, &mut visited, 0, 0, mid) {
                result = mid;
                right = mid;
            } else {
                left = mid + 1;
            }

            visited = vec![vec![false; heights[0].len()]; heights.len()];
        }

        result
    }

    fn dfs_1631(
        heights: &Vec<Vec<i32>>,
        visited: &mut Vec<Vec<bool>>,
        row: usize,
        col: usize,
        effort: i32,
    ) -> bool {
        let m = heights.len();
        let n = heights[0].len();

        if row == m - 1 && col == n - 1 {
            return true;
        }

        visited[row][col] = true;

        let directions = vec![(-1, 0), (1, 0), (0, -1), (0, 1)];

        for dir in directions {
            let new_row = (row as i32 + dir.0) as usize;
            let new_col = (col as i32 + dir.1) as usize;

            if new_row < 0
                || new_row >= m
                || new_col < 0
                || new_col >= n
                || visited[new_row][new_col]
            {
                continue;
            }

            let diff = (heights[new_row][new_col] - heights[row][col]).abs();

            if diff <= effort {
                if Self::dfs_1631(heights, visited, new_row, new_col, effort) {
                    return true;
                }
            }
        }

        false
    }
}

//2312
use std::collections::HashMap;
impl Solution {
    pub fn selling_wood(m: i32, n: i32, prices: Vec<Vec<i32>>) -> i64 {
        let mut dp = vec![vec![0; 201]; 201];

        for p in prices {
            dp[p[0] as usize][p[1] as usize] = p[2] as i64;
        }

        for i in 1..=m {
            for j in 1..=n {
                for k in 1..=(i as usize / 2) {
                    dp[i as usize][j as usize] = dp[i as usize][j as usize]
                        .max(dp[k][j as usize] + dp[i as usize - k][j as usize]);
                }
                for k in 1..=(j as usize / 2) {
                    dp[i as usize][j as usize] = dp[i as usize][j as usize]
                        .max(dp[i as usize][k] + dp[i as usize][j as usize - k]);
                }
            }
        }

        dp[m as usize][n as usize]
    }

    pub fn maximize_profit(m: i32, n: i32, prices: Vec<Vec<i32>>) -> i32 {
        let mut dp = vec![vec![0; (n + 1) as usize]; (m + 1) as usize];

        for price in &prices {
            let hi = price[0] as usize;
            let wi = price[1] as usize;
            let p = price[2];
            dp[hi][wi] = p;
        }
        for i in 1..=m as usize {
            for j in 1..=n as usize {
                for k in 1..=i / 2 as usize {
                    dp[i][j] = dp[i][j].max(dp[i - k][j] + dp[k][j]);
                }
                for k in 1..=j / 2 as usize {
                    dp[i][j] = dp[i][j].max(dp[i][j - k] + dp[i][k]);
                }
            }
        }
        dp[m as usize][n as usize]
    }

    pub fn selling_wood1(m: i32, n: i32, prices: Vec<Vec<i32>>) -> i64 {
        let mut value: HashMap<(i32, i32), i32> = HashMap::new();
        let mut memo: Vec<Vec<i64>> = vec![vec![-1; (n + 1) as usize]; (m + 1) as usize];

        fn dfs(x: i32, y: i32, memo: &mut Vec<Vec<i64>>, value: &HashMap<(i32, i32), i32>) -> i64 {
            if memo[x as usize][y as usize] != -1 {
                return memo[x as usize][y as usize];
            }
            let mut ret = if value.contains_key(&(x, y)) {
                value[&(x, y)] as i64
            } else {
                0
            };
            println!("i:{},j{}", x, y);
            if x > 1 {
                for i in 1..x {
                    ret = ret.max(dfs(i, y, memo, value) + dfs(x - i, y, memo, value));
                }
            }
            if y > 1 {
                for j in 1..y {
                    ret = ret.max(dfs(x, j, memo, value) + dfs(x, y - j, memo, value));
                }
            }
            memo[x as usize][y as usize] = ret;
            println!("{:?}", memo);
            ret
        };

        for price in prices {
            value.insert((price[0], price[1]), price[2]);
        }
        dfs(m, n, &mut memo, &value)
    }
}

//1631
impl Solution {
    pub fn minimum_effort_path_2(heights: Vec<Vec<i32>>) -> i32 {
        let m = heights.len();
        let n = heights[0].len();
        let mut edges = Vec::new();

        for i in 0..m {
            for j in 0..n {
                let index = i * n + j;

                if i > 0 {
                    edges.push((index - n, index, (heights[i][j] - heights[i - 1][j]).abs()));
                }

                if j > 0 {
                    edges.push((index - 1, index, (heights[i][j] - heights[i][j - 1]).abs()));
                }
            }
        }

        edges.sort_by_key(|&(_, _, diff)| diff);

        let mut uf = UnionFind::new(m * n);
        let mut result = 0;

        for (x, y, diff) in edges {
            uf.union(x, y);

            if uf.find(0) == uf.find(m * n - 1) {
                result = diff;
                break;
            }
        }

        result
    }
}

//2864
impl Solution {
    pub fn maximum_odd_binary_number1(s: String) -> String {
        let n = s.len();
        let mut cnt = 0;
        for c in s.chars() {
            if c == '1' {
                cnt += 1;
            }
        }
        cnt -= 1;
        let mut ans = String::new();
        ans.push('1');
        for i in 1..n {
            if i == n - cnt {
                ans.push('1');
                cnt -= 1;
            } else {
                ans.push('0');
            }
        }
        ans.chars().rev().collect()
    }
}

use std::fmt::{self, format};
use std::{i32, iter, vec};
//2864
impl Solution {
    pub fn maximum_odd_binary_number(s: String) -> String {
        let ones = s.as_bytes().iter().filter(|x| **x == b'1').count() - 1;
        iter::repeat('1')
            .take(ones)
            .chain(iter::repeat('0').take(s.len() - ones - 1))
            .chain(iter::once('1'))
            .collect()
    }
}

//821
impl Solution {
    // 方法一：暴力法
    pub fn shortest_to_char_1(s: String, c: char) -> Vec<i32> {
        let mut result: Vec<i32> = vec![0; s.len()];
        let chars: Vec<char> = s.chars().collect();

        for i in 0..s.len() {
            let mut distance = 0;
            while chars[i] != c {
                distance += 1;
                if i + distance < s.len() && chars[i + distance] == c {
                    break;
                }
                if i >= distance && chars[i - distance] == c {
                    break;
                }
            }
            result[i] = distance as i32;
        }

        result
    }

    // 方法二：动态规划
    pub fn shortest_to_char_2(s: String, c: char) -> Vec<i32> {
        let mut result = vec![0; s.len()];
        let chars: Vec<char> = s.chars().collect();
        let mut prev = -10000;

        for i in 0..s.len() {
            if chars[i] == c {
                prev = i as i32;
            }
            result[i] = (i as i32 - prev).abs();
        }

        prev = 10000;
        for i in (0..s.len()).rev() {
            if chars[i] == c {
                prev = i as i32;
            }
            result[i] = result[i].min(prev - i as i32);
        }

        result
    }

    // 方法三：双指针
    pub fn shortest_to_char_3(s: String, c: char) -> Vec<i32> {
        let mut result = vec![0; s.len()];
        let chars: Vec<char> = s.chars().collect();
        let mut left = -10000;
        let mut right = 10000;

        for i in 0..s.len() {
            if chars[i] == c {
                left = i as i32;
            }
            result[i] = (i as i32 - left).abs();
        }

        for i in (0..s.len()).rev() {
            if chars[i] == c {
                right = i as i32;
            }
            result[i] = result[i].min(right - i as i32);
        }

        result
    }

    //方法四：并查集
    pub fn shortest_to_char_4(s: String, c: char) -> Vec<i32> {
        let n = s.len();
        let mut uf = UnionFind::new(n);
        let mut result = vec![0; n];

        for i in 0..n {
            if s.chars().nth(i).unwrap() == c {
                uf.union(i, i);
            }
        }

        for i in 0..n {
            if s.chars().nth(i).unwrap() != c {
                let mut min_dist = n;
                for j in 0..n {
                    if s.chars().nth(j).unwrap() == c {
                        let dist = (i as i32 - j as i32).abs();
                        min_dist = min_dist.min(dist as usize);
                    }
                }
                result[i] = min_dist as i32;
            }
        }

        result
    }
}

// 162
impl Solution {
    pub fn find_peak_element(nums: Vec<i32>) -> i32 {
        let mut left = 0;
        let mut right = nums.len() - 1;

        while left < right {
            let mid = left + (right - left) / 2;
            if nums[mid] < nums[mid + 1] {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        left as i32
    }
}

//3095
impl Solution {
    pub fn minimum_subarray_length(nums: Vec<i32>, k: i32) -> i32 {
        let mut ans = i32::max_value();
        for i in 0..nums.len() {
            let mut tmp = nums[i];
            if tmp >= k {
                ans = 1;
                continue;
            } else {
                let mut j = i + 1;
                while j < nums.len() {
                    tmp = tmp | nums[j];
                    if tmp >= k {
                        ans = ans.min((j - i + 1) as i32);
                        break;
                    } else {
                        j += 1;
                    }
                }
            }
        }
        if ans == i32::max_value() {
            ans = -1;
        }
        ans
    }

    // 方法二：滑动窗口位运算
    pub fn minimum_subarray_length2(nums: Vec<i32>, k: i32) -> i32 {
        let n = nums.len();
        let mut bits = [0; 30];
        let mut res = i32::MAX;

        let calc = |bits: &[i32]| -> i32 {
            let mut ans = 0;
            for i in 0..30 {
                if bits[i] > 0 {
                    ans |= 1 << i;
                }
            }
            ans
        };

        let mut left = 0;
        for right in 0..n {
            for i in 0..30 {
                bits[i] += (nums[right] >> i) & 1;
            }
            while left <= right && calc(&bits) >= k {
                res = res.min((right - left + 1) as i32);
                for i in 0..30 {
                    bits[i] -= (nums[left] >> i) & 1;
                }
                left += 1;
            }
        }

        if res == i32::MAX {
            -1
        } else {
            res
        }
    }
}

impl Solution {
    pub fn minimum_subarray_length3(mut nums: Vec<i32>, k: i32) -> i32 {
        let mut ans = usize::MAX;
        for i in 0..nums.len() {
            let x = nums[i];
            if x >= k {
                return 1;
            }
            let mut j = i - 1;
            while j < nums.len() && (nums[j] | x) != nums[j] {
                nums[j] |= x;
                if nums[j] >= k {
                    ans = ans.min(i - j + 1);
                }
                j -= 1;
            }
        }
        if ans == usize::MAX { -1 } else { ans as _ }
    }
}

/// 2296文本编辑器
#[derive(Debug)]
struct TextEditor {
    text: String,
    cursor: usize,
}


/** 
 * `&self` means the method takes an immutable reference.
 * If you need a mutable reference, change it to `&mut self` instead.
 */
impl TextEditor {
    fn new() -> Self {
        Self{
            text: "".to_string(),
            cursor: 0,
        }
    }
    
    fn add_text(&mut self, text: String) {
        self.text.insert_str(self.cursor, &text);
        self.cursor += text.len();
    }
    
    fn delete_text(&mut self, k: i32) -> i32 {
        let ans = min(self.cursor as i32, k);
        if ans == 0 {
            return 0;
        }
        self.text.drain(self.cursor - ans as usize ..self.cursor);
        self.cursor = max(self.cursor - ans as usize, 0);
        ans
    }
    
    fn cursor_left(&mut self, k: i32) -> String {
        self.cursor = max(self.cursor as i32 - k , 0) as usize;
        self.text[self.cursor - min(self.cursor , 10) ..self.cursor].to_string()
    }
    
    fn cursor_right(&mut self, k: i32) -> String {
        self.cursor = min(self.cursor as i32  + k , self.text.len() as i32) as usize;
        self.text[self.cursor - min(self.cursor , 10) ..self.cursor].to_string()
    }
}

// 解法2 分左右光标
struct TextEditor1 {
    // 光标前面的
    left: Vec<char>,

    // 光标后面的
    right: Vec<char>
}

impl TextEditor1 {
    fn new() -> Self {
        Self {
            left: vec![],
            right: vec![]
        }
    }

    fn add_text(&mut self, text: String) {
        for c in text.chars() {
            self.left.push(c);
        }
    }

    fn delete_text(&mut self, k: i32) -> i32 {
        let len = self.left.len().min(k as usize);
        self.left.truncate(self.left.len() - len);
        len as i32
    }

    fn cursor_left(&mut self, k: i32) -> String {
        let len = self.left.len().min(k as usize);
        for _ in 0..len {
            if let Some(x) = self.left.pop() {
                self.right.push(x);
            }
        }
        let chars  = &self.left[(self.left.len() - 10.min(self.left.len()))..self.left.len()];
        String::from_iter(chars.iter())
    }

    fn cursor_right(&mut self, k: i32) -> String {
        let len = self.right.len().min(k as usize);
        for _ in 0..len {
            if let Some(x) = self.right.pop() {
                self.left.push(x);
            }
        }
        let chars  = &self.left[(self.left.len() - 10.min(self.left.len()))..self.left.len()];
        String::from_iter(chars.iter())
    }
}
/**
 * Your TextEditor object will be instantiated and called as such:
 * let obj = TextEditor::new();
 * obj.add_text(text);
 * let ret_2: i32 = obj.delete_text(k);
 * let ret_3: String = obj.cursor_left(k);
 * let ret_4: String = obj.cursor_right(k);
 */

//3066
impl Solution {
    pub fn min_operations(nums: Vec<i32>, k: i32) -> i32 {
        use core::cmp::Reverse;
        use std::collections::BinaryHeap;
        let mut res = 0;
        let mut pq: BinaryHeap<Reverse<i64>> = BinaryHeap::new();
        for &num in &nums {
            pq.push(Reverse(num as i64));
        }
        while let Some(Reverse(x)) = pq.pop() {
            if x >= k as i64 {
                break;
            }
            if let Some(Reverse(y)) = pq.pop() {
                pq.push(Reverse(x + x + y));
                res += 1;
            }
        }
        res
    }
}

//2684
impl Solution {
    //wrong
    pub fn max_moves(grid: Vec<Vec<i32>>) -> i32 {
        let m = grid.len();
        let n = grid[0].len();
        let mut dp = vec![vec![0; n]; m];
        let mut res = 0;
        // 从右往左遍历每一列
        for j in (0..n).rev() {
            // 从上到下遍历每一行
            for i in 0..m {
                // 计算当前位置能够移动的最大次数
                dp[i][j] = 0;
                if j + 1 < n {
                    if i > 0 && grid[i][j] < grid[i - 1][j + 1] {
                        dp[i][j] = dp[i][j].max(dp[i - 1][j + 1] + 1);
                    }
                    if grid[i][j] < grid[i][j + 1] {
                        dp[i][j] = dp[i][j].max(dp[i][j + 1] + 1);
                    }
                    if i + 1 < m && grid[i][j] < grid[i + 1][j + 1] {
                        dp[i][j] = dp[i][j].max(dp[i + 1][j + 1] + 1);
                    }
                }
                if j == 0 {
                    res = res.max(dp[i][j]);
                }
            }
        }
        println!("dp:{:?}", dp);
        res
        // // 返回dp数组中的最大值
        // dp.iter().map(|row| *row.iter().max().unwrap()).max().unwrap()
    }
}

impl Solution {
    pub fn max_moves1(grid: Vec<Vec<i32>>) -> i32 {
        let mut res = 0;

        fn dfs(g: &Vec<Vec<i32>>, i: usize, j: usize) -> i32 {
            let m = g.len();
            let n = g[0].len();
            if j == n - 1 {
                return 0;
            }
            let mut res = 0;
            if i > 0 && g[i][j] < g[i - 1][j + 1] {
                res = res.max(dfs(&g, i - 1, j + 1) + 1);
            }
            if g[i][j] < g[i][j + 1] {
                res = res.max(dfs(&g, i, j + 1) + 1);
            }
            if i + 1 < m && g[i][j] < g[i + 1][j + 1] {
                res = res.max(dfs(&g, i + 1, j + 1) + 1);
            }
            res
        }

        for i in 0..grid.len() {
            res = res.max(dfs(&grid, i, 0));
            println!("res:{}", res);
        }
        res
    }
}

//2239
impl Solution {
    pub fn find_closest_number(nums: Vec<i32>) -> i32 {
        let mut ans = nums[0];
        for num in nums.into_iter() {
            if ans.abs() > num.abs() {
                ans = num;
            } else if ans.abs() == num.abs() {
                ans = ans.max(num);
            }
        }
        ans
    }
}

//1793
impl Solution {
    //超时
    pub fn maximum_score(nums: Vec<i32>, k: i32) -> i64 {
        use std::cmp::{max, min};
        let n = nums.len();
        let mut ans = 0;
        for i in 0..=k as usize {
            let mut tmp = nums[i] as i64;
            for k in i..=k as usize {
                tmp = min(tmp, nums[k] as i64);
            }
            for j in k as usize..n {
                tmp = min(tmp, nums[j] as i64);
                ans = max(ans, tmp * ((j - i + 1) as i64));
            }
        }
        ans
    }

    //双指针
    pub fn maximum_score_1(nums: Vec<i32>, k: i32) -> i32 {
        let n = nums.len() as i32;
        let mut left = k - 1;
        let mut right = k + 1;
        let mut ans = 0;
        for i in (0..=nums[k as usize]).rev() {
            while left >= 0 && left < n && nums[left as usize] >= i {
                left -= 1;
            }
            while right < n && nums[right as usize] >= i {
                right += 1;
            }
            ans = ans.max((right - left - 1) as i32 * i);
            if left == -1 && right == n {
                break;
            }
        }
        ans
    }

    //优化的双指针
    pub fn maximum_score_2(nums: Vec<i32>, k: i32) -> i32 {
        let n = nums.len() as i32;
        let mut left = k as i32 - 1;
        let mut right = k + 1;
        let mut ans = 0;
        let mut i = nums[k as usize];
        loop {
            while left >= 0 && left < n && nums[left as usize] >= i {
                left -= 1;
            }
            while right < n && nums[right as usize] >= i {
                right += 1;
            }
            ans = ans.max((right - left - 1) * i);
            if left == -1 && right == n {
                break;
            }
            let lval = if left == -1 { -1 } else { nums[left as usize] };
            let rval = if right == n { -1 } else { nums[right as usize] };
            i = lval.max(rval);
            if i == -1 {
                break;
            }
        }
        ans
    }
}

//2671
struct FrequencyTracker {
    freq: HashMap<i32, i32>,
    freq_cnt: HashMap<i32, i32>,
}

impl FrequencyTracker {
    fn new() -> Self {
        FrequencyTracker {
            freq: HashMap::new(),
            freq_cnt: HashMap::new(),
        }
    }

    fn add(&mut self, number: i32) {
        let prev = *self.freq.get(&number).unwrap_or(&0);
        *self.freq_cnt.entry(prev).or_insert(0) -= 1;
        *self.freq.entry(number).or_insert(0) += 1;
        *self.freq_cnt.entry(prev + 1).or_insert(0) += 1;
    }

    fn delete_one(&mut self, number: i32) {
        if self.freq.get(&number).unwrap_or(&0) == &0 {
            return;
        }
        let prev = *self.freq.get(&number).unwrap();
        *self.freq_cnt.entry(prev).or_insert(0) -= 1;
        *self.freq.entry(number).or_insert(0) -= 1;
        *self.freq_cnt.entry(prev - 1).or_insert(0) += 1;
    }

    fn has_frequency(&self, frequency: i32) -> bool {
        self.freq_cnt.get(&frequency).unwrap_or(&0) > &0
    }
}

// 3129 找出所有稳定的二进制数组I
// https://leetcode.cn/problems/find-all-possible-stable-binary-arrays-i/description/
impl Solution {
    pub fn number_of_stable_arrays(zero: i32, one: i32, limit: i32) -> i32 {
        const MOD: i32 = 1_000_000_007;
        let mut dp: Vec<Vec<i32>> = Vec::new();

        todo!()
    }
}

//2266
impl Solution {
    pub fn count_texts(pressed_keys: String) -> i32 {
        let m = 1000000007;
        let n = pressed_keys.len();
        let mut dp3 = vec![1, 1, 2, 4]; // 连续按多次 3 个字母按键对应的方案数
        let mut dp4 = vec![1, 1, 2, 4]; // 连续按多次 4 个字母按键对应的方案数
        for i in 4..=n {
            dp3.push((dp3[i - 1] + dp3[i - 2] + dp3[i - 3]) % m);
            dp4.push((dp4[i - 1] + dp4[i - 2] + dp4[i - 3] + dp4[i - 4]) % m);
        }
        let mut res = 1i64; // 总方案数
        let mut cnt = 1; // 当前字符连续出现的次数
        let pressed_keys: Vec<char> = pressed_keys.chars().collect();
        for i in 1..n {
            if pressed_keys[i] == pressed_keys[i - 1] {
                cnt += 1;
            } else {
                // 对按键对应字符数量讨论并更新总方案数
                if pressed_keys[i - 1] == '7' || pressed_keys[i - 1] == '9' {
                    res = (res * dp4[cnt]) % m as i64;
                } else {
                    res = (res * dp3[cnt]) % m as i64;
                }
                cnt = 1;
            }
        }
        // 更新最后一段连续字符子串对应的方案数
        if pressed_keys[n - 1] == '7' || pressed_keys[n - 1] == '9' {
            res = (res * dp4[cnt]) % m as i64;
        } else {
            res = (res * dp3[cnt]) % m as i64;
        }
        res as i32
    }
}

impl Solution {
    pub fn max_energy_boost(energy_drink_a: Vec<i32>, energy_drink_b: Vec<i32>) -> i64 {
        let n = energy_drink_a.len();
        let mut d = vec![vec![0; 2]; n + 1];
        for i in 1..=n {
            d[i][0] = d[i - 1][0] + energy_drink_a[i - 1] as i64;
            d[i][1] = d[i - 1][1] + energy_drink_b[i - 1] as i64;
            if i >= 2 {
                d[i][0] = d[i][0].max(d[i - 2][1] + energy_drink_a[i - 1] as i64);
                d[i][1] = d[i][1].max(d[i - 2][0] + energy_drink_b[i - 1] as i64);
            }
        }
        d[n][0].max(d[n][1])
    }
}

//70
impl Solution {
    pub fn climb_stairs(n: i32) -> i32 {
        let mut dp = vec![1,1,2];
        for i in 3..=n as usize {
            dp.push(dp[i-1] + dp[i-2]);
        }
        dp[n as usize]
    }
}

//2218
impl Solution {
    pub fn max_value_of_coins(piles: Vec<Vec<i32>>, k: i32) -> i32 {
        let mut f = vec![-1; (k + 1) as usize];
        f[0] = 0;
        for pile in piles {
            for i in (1..= k as usize).rev() {
                let mut value = 0;
                for t in 1..=pile.len() {
                    value += pile[t - 1];
                    if i >= t && f[i - t] != -1 {
                        f[i] = f[i].max(f[i - t] + value);
                    }
                }
            }
        }
        f[k as usize]
    }
}

// 2595
impl Solution {
    pub fn even_odd_bit(n: i32) -> Vec<i32> {
        let n_binary = format!("{:b}", n);
        let mut ans1 = 0;
        let mut ans2 = 0;
        println!("n binary: {:?}", n_binary);
        for (i, c) in n_binary.chars().rev().enumerate() {
            if c == '1' {
                if i % 2 == 0 {
                    ans1 += 1;
                } else {
                    ans2 += 1;
                }
            }
        }
        vec![ans1, ans2]
    }
}

//2353  方法一超时
struct FoodRatings {
    foods: HashMap<String, (String, i32)>, // 食物名 -> (类别, 评分)
    categories: HashMap<String, Vec<String>>, // 类别 -> 食物名列表
}

impl FoodRatings {
    fn new(foods: Vec<String>, categories: Vec<String>, ratings: Vec<i32>) -> Self {
        let mut food_ratings = HashMap::new();
        let mut categories_map = HashMap::new();

        for (i, food) in foods.iter().enumerate() {
            let category = &categories[i];
            let rating = ratings[i];
            food_ratings.insert(food.clone(), (category.clone(), rating));

            categories_map
                .entry(category.clone())
                .or_insert(Vec::new())
                .push(food.clone());
        }

        FoodRatings {
            foods: food_ratings,
            categories: categories_map,
        }
    }

    fn highest_rated(&self, category: String) -> String {
        let mut max_rating_food = None;

        if let Some(food_list) = self.categories.get(&category) {
            for food in food_list {
                if let Some((_, rating)) = self.foods.get(food) {
                    if max_rating_food.is_none()
                        || max_rating_food
                            .as_ref()
                            .map_or(true, |(_, max_rating)| rating > max_rating)
                    {
                        max_rating_food = Some((food.clone(), *rating));
                    } else if max_rating_food
                        .as_ref()
                        .map_or(false, |(max_food, cur_rating)| *rating == *cur_rating)
                        && food < &max_rating_food.as_ref().unwrap().0
                    {
                        max_rating_food = Some((food.clone(), *rating));
                    }
                }
            }
        }

        max_rating_food.unwrap_or_default().0
    }

    fn change_rating(&mut self, food: String, new_rating: i32) {
        if let Some((_, rating)) = self.foods.get_mut(&food) {
            *rating = new_rating;
        }
    }
}

//方法二
use std::collections::BTreeSet;

struct FoodRatings2 {
    food_map: HashMap<String, (i32, String)>,
    rating_map: HashMap<String, BTreeSet<(i32, String)>>,
    n: usize,
}

impl FoodRatings2 {

    fn new(foods: Vec<String>, cuisines: Vec<String>, ratings: Vec<i32>) -> Self {
        let n = foods.len();
        let mut food_map = HashMap::new();
        let mut rating_map = HashMap::new();

        for i in 0..n {
            let food = foods[i].clone();
            let cuisine = cuisines[i].clone();
            let rating = ratings[i];
            food_map.insert(food.clone(), (rating, cuisine.clone()));
            rating_map
                .entry(cuisine)
                .or_insert_with(BTreeSet::new)
                .insert((n as i32 - rating, food));
        }

        Self {
            food_map,
            rating_map,
            n,
        }
    }
    
    fn change_rating(&mut self, food: String, new_rating: i32) {
        if let Some((old_rating, cuisine)) = self.food_map.get(&food) {
            let old_rating = *old_rating;
            let cuisine = cuisine.clone();
            self.rating_map
                .get_mut(&cuisine)
                .unwrap()
                .remove(&(self.n as i32 - old_rating, food.clone()));
            self.rating_map
                .get_mut(&cuisine)
                .unwrap()
                .insert((self.n as i32 - new_rating, food.clone()));
            self.food_map.insert(food, (new_rating, cuisine));
        }
    }
    
    fn highest_rated(&self, cuisine: String) -> String {
        self.rating_map
            .get(&cuisine)
            .and_then(|set| set.iter().next())
            .map(|(_, food)| food.clone())
            .unwrap()
    }
}

//方法三
use std::collections::BinaryHeap;
use std::cmp::Ordering;

#[derive(Debug, Eq, PartialEq)]
struct Pair(i32, String);

impl PartialOrd for Pair {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Pair {
    fn cmp(&self, other: &Self) -> Ordering {
        other.0.cmp(&self.0).then_with(|| other.1.cmp(&self.1))
    }
}

#[derive(Debug)]
struct FoodRatings3 {
    food_map: HashMap<String, (i32, String)>,
    rating_map: HashMap<String, BinaryHeap<Pair>>,
    n: i32,
}

impl FoodRatings3 {
    fn new(foods: Vec<String>, cuisines: Vec<String>, ratings: Vec<i32>) -> Self {
        let n = foods.len() as i32;
        let mut food_map = HashMap::new();
        let mut rating_map = HashMap::new();

        for i in 0..foods.len() {
            let food = foods[i].clone();
            let cuisine = cuisines[i].clone();
            let rating = ratings[i];
            food_map.insert(food.clone(), (rating, cuisine.clone()));
            rating_map
                .entry(cuisine.clone())
                .or_insert_with(BinaryHeap::new)
                .push(Pair(n - rating, food.clone()));
        }

        FoodRatings3 {
            food_map,
            rating_map,
            n,
        }
    }
    
    fn change_rating(&mut self, food: String, new_rating: i32) {
        if let Some((old_rating, cuisine)) = self.food_map.get_mut(&food) {
            *old_rating = new_rating;
            self.rating_map
                .get_mut(cuisine)
                .unwrap()
                .push(Pair(self.n - new_rating, food.clone()));
        }
    }
    
    fn highest_rated(&mut self, cuisine: String) -> String {
        if let Some(heap) = self.rating_map.get_mut(&cuisine) {
            while let Some(&ref item) = heap.peek() {
                let rating = &item.0;
                let food = &item.1;
                if self.n - rating == self.food_map[food].0 {
                    return food.clone();
                }
                heap.pop();
            }
        }
        String::new()
    }
}
/**
 * Your FoodRatings object will be instantiated and called as such:
 * let obj = FoodRatings::new(foods, cuisines, ratings);
 * obj.change_rating(food, newRating);
 * let ret_2: String = obj.highest_rated(cuisine);
 */

 //131分割回文串 I
 use std::collections::VecDeque;

impl Solution {
    pub fn partition(s: String) -> Vec<Vec<String>> {
        let n = s.len();
        let mut res = vec![];
        let mut stack = VecDeque::from([(0, vec![])]);

        println!("初始 stack: {:?}", stack);

        while let Some((start, acc)) = stack.pop_back() {
            println!("弹出: start = {}, acc = {:?}", start, acc);
            
            for end in start + 1..=n {
                println!("   (start={}, end={})",  start, end);
                let t = (&s[start..end]).to_string();
                if t.chars().eq(t.chars().rev()) {
                    let mut acc = acc.clone();
                    acc.push(t.clone());
                    
                    println!("  发现回文: {:?} (start={}, end={})", t, start, end);
                    
                    if end == n {
                        println!("  🔥 结果加入: {:?}", acc);
                        res.push(acc);
                    } else {
                        println!("  压入 stack: (start={}, acc={:?})", end, acc);
                        stack.push_back((end, acc));
                    }
                }
            }
            println!("当前 stack: {:?}", stack);
        }

        println!("最终结果: {:?}", res);
        res
    }
}

//132分割回文串 II
impl Solution {
    // 超时
    pub fn min_cut(s: String) -> i32 {
        let mut ans = i32::MAX;
        let n = s.len();
        let mut stack = VecDeque::from([(0, vec![])]);
        while let Some((start, acc)) = stack.pop_back() {
            for end in start + 1..=n {
                let t = (&s[start..end]).to_string();
                if t.chars().eq(t.chars().rev()){
                    let mut acc = acc.clone();
                    acc.push(t.clone());
                    if end == n {
                        ans = ans.min(acc.len() as i32);
                    } else {
                        stack.push_back((end, acc));
                    }
                }
            }
        }

        ans
    }

    pub fn min_cut2(s: String) -> i32 {
        let n = s.len();
        let s_bytes = s.as_bytes();
        let mut f = vec![vec![true; n]; n];
        for i in (0..n-1).rev() {
            for j in i+1..n {
                f[i][j] = s_bytes[i] == s_bytes[j] && f[i+1][j-1];
            }
        }
        let mut dp = vec![0; n];
        for r in 0..n {
            if f[0][r] {
                continue;
            }
            let mut res = i32::MAX;
            for l in 0..=r {
                if f[l][r] {
                    res = res.min(dp[l - 1] + 1);
                }
            }
            dp[r] = res;
        }

        dp[n-1]
    }

    //最快，中心扩展
    pub fn min_cut3(s: String) -> i32 {
        let s = s.into_bytes();
        let n = s.len();
        let mut f = vec![i32::MAX; n + 1];
        f[0] = 0;
        
        println!("初始 f: {:?}", f);

        for i in 0..n {
            f[i + 1] = f[i + 1].min(f[i] + 1);
            println!("\n--- 处理 i = {} (字符 '{}') ---", i, s[i] as char);
            println!("假设单独切割: f[{}] = min({}, {}) = {}", i + 1, f[i + 1], f[i] + 1, f[i + 1]);

            // 处理奇数长度回文
            println!("  尝试奇数长度回文...");
            for j in 1..=i.min(n - i) {
                if s[i - j] == s[i + j - 1] {
                    f[i + j] = f[i + j].min(f[i - j] + 1);
                    println!("    回文: '{}' (索引 {}-{})", 
                        String::from_utf8_lossy(&s[i - j..i + j]),
                        i - j, i + j - 1
                    );
                    println!("    更新 f[{}]: min({}, {}) = {}", i + j, f[i + j], f[i - j] + 1, f[i + j]);
                } else {
                    println!("    终止扩展: '{}' ≠ '{}'", s[i - j] as char, s[i + j - 1] as char);
                    break;
                }
            }

            // 处理偶数长度回文
            println!("  尝试偶数长度回文...");
            for j in 1..=i.min(n - i - 1) {
                if s[i - j] == s[i + j] {
                    f[i + j + 1] = f[i + j + 1].min(f[i - j] + 1);
                    println!("    回文: '{}' (索引 {}-{})",
                        String::from_utf8_lossy(&s[i - j..i + j + 1]),
                        i - j, i + j
                    );
                    println!("    更新 f[{}]: min({}, {}) = {}", i + j + 1, f[i + j + 1], f[i - j] + 1, f[i + j + 1]);
                } else {
                    println!("    终止扩展: '{}' ≠ '{}'", s[i - j] as char, s[i + j] as char);
                    break;
                }
            }

            println!("  f 状态: {:?}", f);
        }

        println!("\n最终 f: {:?}", f);
        f[n] - 1
    }
}

//1278分割回文串 III
impl Solution {
    pub fn palindrome_partition(s: String, k: i32) -> i32 {
        
        0
    }
}

//3305 元音辅音字符串计数 I
use std::collections::HashSet;
impl Solution {
    pub fn count_of_substrings(word: String, k: i32) -> i32 {
        let vowels = HashSet::from(['a', 'e', 'i', 'o', 'u']);
        let n = word.len();
        let mut res = 0;
        for i in 0..n {
            let mut occur = HashSet::new();
            let mut consonants = 0;
            for j in i..n {
                if vowels.contains(&word[j..j+1].chars().next().unwrap()) {
                    occur.insert(word[j..j+1].chars().next().unwrap());
                } else {
                    consonants += 1;
                }
                if occur.len() == 5 && consonants == k {
                    res += 1;
                }
            }
        }
        res
    }

    //滑动窗口
    pub fn count_of_substrings2(word: String, k: i32) -> i64 {
        let vowels: HashSet<char> = ['a', 'e', 'i', 'o', 'u'].iter().cloned().collect();
        fn count(word: &str, k: i32, vowels: &HashSet<char>) -> i64 {
            let n = word.len();
            let mut res = 0;
            let mut consonants = 0;
            let mut occur: HashMap<char, i32> = HashMap::new();
            let mut j = 0;
            let word_chars: Vec<char> = word.chars().collect();
            for i in 0..n {
                while j < n && (consonants < k || occur.len() < 5) {
                    let ch = word_chars[j];
                    if vowels.contains(&ch) {
                        *occur.entry(ch).or_insert(0) += 1;
                    } else {
                        consonants += 1;
                    }
                    j += 1;
                }
                if consonants >= k && occur.len() == 5 {
                    res += (n - j + 1) as i64;
                }
                let left = word_chars[i];
                if vowels.contains(&left) {
                    if let Some(count) = occur.get_mut(&left) {
                        *count -= 1;
                        if *count == 0 {
                            occur.remove(&left);
                        }
                    }
                } else {
                    consonants -= 1;
                }
            }
            res
        }
        count(&word, k, &vowels) - count(&word, k + 1, &vowels)
    }
}

//字符串处理，域名按数字合并。 tokyo001, shanghai06, shanghai03, shanghai04, shanghai02, 合并得到shanghai02-04,shanghai06,tokyo001 

impl Solution {
    pub fn test_fei_tu(inputs: Vec<&str>) -> Vec<String> {
        let mut ans = Vec::new();
        let mut hash: HashMap<String, Vec<i32>> = HashMap::new();

        for input in inputs {
            let mut name = String::new();
            let mut num_str = String::new();

            for c in input.chars() {
                if c.is_numeric() {
                    num_str.push(c);
                } else {
                    name.push(c);
                }
            }

            if let Ok(num) = num_str.parse::<i32>() {
                hash.entry(name).or_insert(vec![]).push(num);
            }
        }

        for (name, mut nums) in hash {
            nums.sort();

            let mut i = 0;
            let mut parts = Vec::new();

            while i < nums.len() {
                let start = nums[i];
                let mut end = start;

                while i + 1 < nums.len() && nums[i + 1] == end + 1 {
                    i += 1;
                    end = nums[i];
                }

                if start == end {
                    parts.push(format!("{}{:03}", name, start));
                } else {
                    parts.push(format!("{}{:03}-{:03}", name, start, end));
                }

                i += 1;
            }

            ans.extend(parts);
        }

        ans.sort();
        println!("{:?}", ans);
        ans
    }
}

//2278. 字母在字符串中的百分比
impl Solution {
    pub fn percentage_letter(s: String, letter: char) -> i32 {
        let mut ans = 0;
        for i in s.chars() {
            if i == letter {
                ans += 1;
            }
        }
        ans * 100 / s.len() as i32
    }
}

impl Solution {
    pub fn test_code() -> i32 {
        todo!()

    }
}

#[cfg(test)]
mod tests {
    use core::{f32, panic};
    use std::u64;

    use super::*;

    #[test]
    fn test_code() {
        let s = "abcba";
        let rusult = Solution::test_code();
    }


    #[test]
    fn test_132() {
        let s = "abcba";
        let rusult = Solution::min_cut3(s.to_string());
    }

    #[test]
    fn test_131() {
        let s = "aab";
        let rusult = Solution::partition(s.to_string());
    }

    #[test]
    fn test_2353() {
        let mut food_ratings = FoodRatings::new(
            vec!["kimchi".to_string(), "miso".to_string(), "sushi".to_string(), "moussaka".to_string(), "ramen".to_string(), "bulgogi".to_string()],
            vec!["korean".to_string(), "japanese".to_string(), "japanese".to_string(), "greek".to_string(), "japanese".to_string(), "korean".to_string()],
            vec![9, 12, 8, 15, 14, 7]
        );

        // Test highestRated for "korean"
        assert_eq!(food_ratings.highest_rated("korean".to_string()), "kimchi");

        // Test highestRated for "japanese"
        assert_eq!(food_ratings.highest_rated("japanese".to_string()), "ramen");

        // Change rating for "sushi"
        food_ratings.change_rating("sushi".to_string(), 16);

        // Test highestRated for "japanese" after rating change
        assert_eq!(food_ratings.highest_rated("japanese".to_string()), "sushi");

        // Change rating for "ramen"
        food_ratings.change_rating("ramen".to_string(), 16);

        // Test highestRated for "japanese" after both ratings are equal
        assert_eq!(food_ratings.highest_rated("japanese".to_string()), "ramen");
    }

    #[test]
    fn test_2296() {
        let mut editor = TextEditor::new();

        // Test case
        editor.add_text("leetcode".to_string()); // null
        println!("After addText('leetcode'): {}", editor.text); // "leetcode"
    
        let deleted = editor.delete_text(4); // 4
        println!("After deleteText(4): deleted {} chars", deleted); // "4"
        println!("#####editpr:{:?}", editor);
        editor.add_text("practice".to_string()); // null
        println!("After addText('practice'): {}", editor.text); // "lepractice"
    
        let cursor_right = editor.cursor_right(3); // "etpractice"
        println!("After cursorRight(3): {}", cursor_right); // "etpractice"
        println!("#####editpr:{:?}", editor);
        let cursor_left = editor.cursor_left(8); // "leet"
        println!("After cursorLeft(8): {}", cursor_left); // "leet"
    
        let deleted_again = editor.delete_text(10); // 4
        println!("After deleteText(10): deleted {} chars", deleted_again); // "4"
    
        let cursor_left_again = editor.cursor_left(2); // ""
        println!("After cursorLeft(2): {}", cursor_left_again); // ""
    
        let cursor_right_again = editor.cursor_right(6); // "practi"
        println!("After cursorRight(6): {}", cursor_right_again); // "practi"
        panic!()
    }

    #[test]
    fn test_2266() {
        let s = String::from("22");
        let result = Solution::count_texts(s);
        assert_eq!(result, 2);
    }

    #[test]
    fn test_2595() {
        let s = 50;
        let result = Solution::even_odd_bit(s);
    }

    #[test]
    fn test_3095() {
        // let s = vec![1,2,3];
        // let result = Solution::minimum_subarray_length(s, 2);
        // assert_eq!(result, 1);
        let s = vec![2, 1, 8];
        let result = Solution::minimum_subarray_length(s, 10);
        assert_eq!(result, 3);
    }

    #[test]
    fn test_feitu() {
        let s = vec!["shanghai01", "shanghai02", "shanghai04", "shanghai03", "hangzhou03", "hangzhou01"];
        let rusult = Solution::test_fei_tu(s);
        panic!();
    }

    #[test]
    fn test_3066() {
        let s = vec![61, 8, 39, 89, 97, 79, 64, 6];
        let result = Solution::min_operations(s, 98);
        assert_eq!(result, 5);
    }

    #[test]
    fn test_312111() {
        let max1 = f32::MAX;
        let min1 = f32::MIN;
        println!("max1{:?}", max1);
        println!("min1{:?}", min1);
        let u8_1 = max1 as u64;
        println!("{:?}", u8_1);
        let u8_1 = min1 as u64;
        println!("{:?}", u8_1);
        let u64_max: f32 = 1.67;
        let u8_1 = u64_max as u64;
        println!("{:?}", u8_1);
        let i8_1 = u8_1 as i32;
        println!("{:?}", i8_1);
    }

    #[test]
    fn test_3129() {
        let result = Solution::number_of_stable_arrays(1, 1, 2);
        assert_eq!(result, 2);
        let result = Solution::number_of_stable_arrays(3, 3, 2);
        assert_eq!(result, 14);
    }

    #[test]
    fn test_1793() {
        let s = vec![1, 4, 3, 7, 4, 5];
        let result = Solution::maximum_score_2(s, 3);
        assert_eq!(result, 15);
        let s = vec![5, 5, 4, 5, 4, 1, 1, 1];
        let result = Solution::maximum_score(s, 0);
        assert_eq!(result, 20);
    }

    #[test]
    fn test_2671() {
        let mut tracker = FrequencyTracker::new();
        println!("{}", tracker.has_frequency(1));
        tracker.add(1);
        tracker.delete_one(1);
        println!("{}", tracker.has_frequency(1));
    }

    #[test]
    fn max_array_value() {
        let s = vec![2, 3, 7, 9, 3];
        let result = Solution::max_array_value(s);
        assert_eq!(result, 21);
    }

    #[test]
    fn test_2684() {
        let s = vec![vec![3, 2, 4], vec![2, 1, 9], vec![1, 1, 7]];
        let result = Solution::max_moves(s);
        assert_eq!(result, 3);
    }

    #[test]
    fn test_2312() {
        /*
        let m1 = 3;
        let n1 = 5;
        let prices1 = vec![vec![1, 4, 2], vec![2, 2, 7], vec![2, 1, 3]];
        let result = Solution::maximize_profit(m1,n1,prices1);
        assert_eq!(result, 19);*/

        let m1 = 4;
        let n1 = 6;
        let prices1 = vec![vec![3, 2, 10], vec![1, 4, 2], vec![4, 1, 3]];
        let result = Solution::maximize_profit(m1, n1, prices1);
        assert_eq!(result, 32);
    }

    #[test]
    fn test_shortest_to_char_1() {
        let s = String::from("loveleetcode");
        let c = 'e';
        let result = Solution::shortest_to_char_1(s.clone(), c);
        assert_eq!(result, vec![3, 2, 1, 0, 1, 0, 0, 1, 2, 2, 1, 0]);
    }

    #[test]
    fn test_shortest_to_char_2() {
        let s = String::from("loveleetcode");
        let c = 'e';
        let result = Solution::shortest_to_char_2(s.clone(), c);
        assert_eq!(result, vec![3, 2, 1, 0, 1, 0, 0, 1, 2, 2, 1, 0]);
    }

    #[test]
    fn test_shortest_to_char_3() {
        let s = String::from("loveleetcode");
        let c = 'e';
        let result = Solution::shortest_to_char_3(s.clone(), c);
        assert_eq!(result, vec![3, 2, 1, 0, 1, 0, 0, 1, 2, 2, 1, 0]);
    }

    #[test]
    fn test_shortest_to_char_4() {
        let s = String::from("loveleetcode");
        let c = 'e';
        let result = Solution::shortest_to_char_4(s.clone(), c);
        assert_eq!(result, vec![3, 2, 1, 0, 1, 0, 0, 1, 2, 2, 1, 0]);
    }

    #[test]
    fn test_minimum_effort_path() {
        let heights = vec![vec![1, 2, 2], vec![3, 8, 2], vec![5, 3, 5]];
        let result = Solution::minimum_effort_path_1(heights.clone());
        let result2 = Solution::minimum_effort_path_2(heights.clone());
        assert_eq!(result, 2);
        assert_eq!(result2, 2);
    }

    #[test]
    fn test_is_palindrome() {
        let s = "0P".to_string();
        let result = Solution::is_palindrome(s.clone());
        assert_eq!(result, false);
    }

    #[test]
    fn test_single_number() {
        let nums = vec![4, 1, 2, 1, 2, 4, 5];
        let result = Solution::single_number(nums.clone());
        assert_eq!(result, 5);
    }

    #[test]
    fn test_count_pairs() {
        let target = -2;
        let nums = vec![-6, 2, 5, -2, -7, -1, 3];
        let result = Solution::count_pairs(nums.clone(), target);
        assert_eq!(result, 10);
    }

    #[test]
    fn test_convert_to_title() {
        let target = 53;
        let result = Solution::convert_to_title(target);
        assert_eq!(result, "BA".to_string());
    }

    #[test]
    fn test_majority_element() {
        let nums = vec![2, 2, 1, 1, 1, 2, 2];
        let result = Solution::majority_element_1(nums.clone());
        let result2 = Solution::majority_element_2(nums.clone());
        assert_eq!(result, 2);
        assert_eq!(result2, 2);
    }

    #[test]
    fn test_sum_subarray_mins() {
        // Test case 1
        let arr1 = vec![3, 1, 2, 4];
        assert_eq!(Solution::sum_subarray_mins(arr1), 17);

        // Test case 2
        let arr2 = vec![11, 81, 94, 43, 3];
        assert_eq!(Solution::sum_subarray_mins(arr2), 444);
    }

    #[test]
    fn test_find_peak_element() {
        let nums = vec![1, 2, 1, 3, 5, 6, 4];
        let result = Solution::find_peak_element(nums);
        assert_eq!(result, 5);
    }

    #[test]
    fn test_maximum() {
        let s2 = "111".to_string();
        let result = Solution::maximum_odd_binary_number(s2);
        assert_eq!(result, "111".to_string());
    }
}

fn main() {
    println!("Hello, world!");
}
