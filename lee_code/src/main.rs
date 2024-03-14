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

struct Solution;
//125
impl Solution {
    pub fn is_palindrome(s: String) -> bool {
        let clean_str:String = s.chars().filter(|x| x.is_ascii_alphanumeric()).collect();
        let rever_str:String = clean_str.chars().rev().collect();
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

//2824
impl Solution {
    pub fn count_pairs(nums: Vec<i32>, target: i32) -> i32 {
        let mut ans = 0;
        let n = nums.len();
        for i in 0..n - 1 {
            let a = nums[i];
            for j in i+1..n {
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
        use std::{collections::HashMap, arch::x86_64};
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
    
    fn dfs_1631(heights: &Vec<Vec<i32>>, visited: &mut Vec<Vec<bool>>, row: usize, col: usize, effort: i32) -> bool {
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
            
            if new_row < 0 || new_row >= m || new_col < 0 || new_col >= n || visited[new_row][new_col] {
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
        cnt -=1;
        let mut ans = String::new();
        ans.push('1');
        for i in 1..n {
            if i == n-cnt {
                ans.push('1');
                cnt -= 1;
            }
            else{
                ans.push('0');
            }
        }
        ans.chars().rev().collect()
    }
}

use std::iter;
//2864
impl Solution {
    pub fn maximum_odd_binary_number(s: String) -> String {
        let ones=s.as_bytes().iter().filter(|x|**x==b'1').count()-1;
        iter::repeat('1').take(ones).chain(iter::repeat('0').take(s.len()-ones-1)).chain(iter::once('1')).collect()
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

#[cfg(test)]
mod tests {
    use super::*;

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
        let heights = vec![
            vec![1, 2, 2],
            vec![3, 8, 2],
            vec![5, 3, 5]
        ];
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
        let nums = vec![4,1,2,1,2,4,5];
        let result = Solution::single_number(nums.clone());
        assert_eq!(result, 5);
    }

    #[test]
    fn test_count_pairs() {
        let target = -2;
        let nums = vec![-6,2,5,-2,-7,-1,3];
        let result = Solution::count_pairs(nums.clone() , target);
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
        let nums = vec![2,2,1,1,1,2,2];
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
        let arr2 = vec![11,81,94,43,3];
        assert_eq!(Solution::sum_subarray_mins(arr2), 444);
    }

    #[test]
    fn test_find_peak_element() {
        let nums = vec![1, 2, 1, 3, 5, 6 ,4];
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