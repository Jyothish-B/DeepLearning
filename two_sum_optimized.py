# Optimized Two Sum Solution in Python

## Multiple Approaches

### 1. Brute Force Approach
This is the simplest approach where we check every pair of numbers to see if they sum up to the target.

**Time Complexity:** O(n^2)

```python
def two_sum_brute_force(nums, target):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []
```

### 2. Hash Map Approach
This approach uses a hash map to store the numbers and their indices. This allows us to find the complement of each number in constant time.

**Time Complexity:** O(n)

```python
def two_sum_hash_map(nums, target):
    num_map = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_map:
            return [num_map[complement], i]
        num_map[num] = i
    return []
```

### 3. Sorting and Two Pointers Approach
In this approach, we sort the array and use two pointers to find the two numbers.

**Time Complexity:** O(n log n)

```python
def two_sum_two_pointers(nums, target):
    nums_sorted = sorted((num, i) for i, num in enumerate(nums))
    left, right = 0, len(nums_sorted) - 1
    while left < right:
        current_sum = nums_sorted[left][0] + nums_sorted[right][0]
        if current_sum == target:
            return [nums_sorted[left][1], nums_sorted[right][1]]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    return []
```

## Time Complexity Analysis
- **Brute Force Approach:** O(n^2)
- **Hash Map Approach:** O(n)
- **Sorting and Two Pointers Approach:** O(n log n)

## Comprehensive Examples

1. **Example 1:**
   - **Input:** `nums = [2, 7, 11, 15], target = 9`
   - **Output:** `[0, 1]` (because `nums[0] + nums[1] = 2 + 7 = 9`)

2. **Example 2:**
   - **Input:** `nums = [3, 2, 4], target = 6`
   - **Output:** `[1, 2]` (because `nums[1] + nums[2] = 2 + 4 = 6`)

3. **Example 3:**
   - **Input:** `nums = [3, 3], target = 6`
   - **Output:** `[0, 1]` (because `nums[0] + nums[1] = 3 + 3 = 6`)

## Conclusion
The two-sum problem can be solved using various methods, each with its own advantages and trade-offs. The hash map approach is generally the most optimal for performance in terms of time complexity.