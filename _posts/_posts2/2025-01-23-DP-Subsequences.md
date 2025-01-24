---
layout: post
title: DP on Subsequences (Patterns and Problems)
date: 2025-01-20 13:52:00-0400
featured: false
description: A structured reference set of DP on Subsequences problems, directly adapted from Striver's DP playlist.
tags: DP Code
categories: DSA
giscus_comments: false
related_posts: false
# toc:
#   sidebar: left
---


#### **Problems:**

[Subset Sum Problem](https://www.geeksforgeeks.org/problems/subset-sum-problem-1611555638/1)
```c++
//{ Driver Code Starts

#include <bits/stdc++.h>
using namespace std;


// } Driver Code Ends
// User function template for C++

class Solution {
  public:
    // int fn(int idx, int target, vector<int>& arr, vector<vector<int>>& dp) {
    //     if(target==0) return true;
    //     if(idx==0) return (arr[0]==target);
    //     if(dp[idx][target]!=-1) return dp[idx][target];
    //     bool not_take = fn(idx-1, target, arr, dp);
    //     bool take = false;
    //     if(arr[idx]<=target) {
    //         take = fn(idx-1, target-arr[idx], arr, dp);
    //     }
        
    //     return dp[idx][target] = take||not_take;
    // }
    bool isSubsetSum(vector<int>& arr, int target) {
        int n = arr.size();
        //Memoization:
        // vector<vector<int>> dp(n, vector<int>(target+1, -1));
        // return fn(n-1, target, arr, dp)==1;
        
        //Tabulation:
        // vector<vector<bool>> dp(n, vector<bool>(target+1, 0));
        // //Base cases;
        // for(int i=0; i<n; i++) dp[i][0]=true;
        // dp[0][arr[0]] = true;
        // //Building the steps;
        // for(int i=1; i<n; i++) {
        //     for(int j=1; j<=target; j++) {
        //         bool not_take = dp[i-1][j];
        //         bool take = false;
        //         if(arr[i]<=j) {
        //             take = dp[i-1][j-arr[i]];
        //         }
                
        //         dp[i][j] = take||not_take;
        //     }
        // }
        
        // return dp[n-1][target];
        
        //Tabulation + Space Optimization;
        vector<bool> prev(target+1, 0), cur(target+1, 0);
        //Base cases;
        prev[0] = true;
        prev[arr[0]] = true;
        //Building the steps;
        for(int i=1; i<n; i++) {
            cur[0] = true;
            for(int j=1; j<=target; j++) {
                bool not_take = prev[j];
                bool take = false;
                if(arr[i]<=j) {
                    take = prev[j-arr[i]];
                }
                cur[j] = take||not_take;
            }
            prev = cur;
        }
        return prev[target];
    }
};
```
**Key:** Use DP to solve it by deciding for each element whether to include it in the subset or not. Base cases handle the sum being 0 (always true) or only one element matching the target.


[Partition Equal Subset Sum](https://leetcode.com/problems/partition-equal-subset-sum/)
```c++
class Solution {
public:
    // int fn(int idx, int target, vector<int>& nums, vector<vector<int>>& dp) {
    //     if(target==0) return true;
    //     if(idx==0) return (nums[0]==target);
    //     if(dp[idx][target]!=-1) return dp[idx][target];
    //     bool not_take = fn(idx-1, target, nums, dp);
    //     bool take = false;
    //     if(nums[idx]<=target) {
    //         take = fn(idx-1, target-nums[idx], nums, dp);
    //     }
    //     return dp[idx][target] = take||not_take;
    // }
    bool canPartition(vector<int>& nums) {
        int n = nums.size();
        int totSum = 0;
        for(int i=0; i<n; i++) totSum+=nums[i];
        if(totSum%2!=0) return false;
        int target = totSum/2;

        //Memoization:
        // vector<vector<int>> dp(n, vector<int>(target+1, -1));
        // return fn(n-1, target, nums, dp);

        //Tabulation:
        // vector<vector<bool>> dp(n, vector<bool>(target+1, 0));
        // //Base case;
        // for(int i=0; i<n; i++) dp[i][0] = true;
        // if(nums[0]<=target) dp[0][nums[0]] = true;
        // //Building the steps;
        // for(int i=1; i<n; i++) {
        //     for(int j=1; j<=target; j++) {
        //         bool not_take = dp[i-1][j];
        //         bool take = false;
        //         if(nums[i]<=j) {
        //             take = dp[i-1][j-nums[i]];
        //         }
        //         dp[i][j] = take||not_take;
        //     }
        // }
        // return dp[n-1][target]==1;

        //Tabulation + Space Optimization:
        vector<bool> prev(target+1, 0), cur(target+1, 0);
        //Base case;
        prev[0] = true;
        if(nums[0]<=target) prev[nums[0]] = true;
        //Building the steps;
        for(int i=1; i<n; i++) {
            cur[0] = true;
            for(int j=1; j<=target; j++) {
                bool not_take = prev[j];
                bool take = false;
                if(nums[i]<=j) {
                    take = prev[j-nums[i]];
                }
                cur[j] = take||not_take;
            }
            prev = cur;
        }
        return prev[target];
    }
};
```
**Key:** Check if the array can be split into two subsets with equal sum by solving a subset sum problem for half the total sum using DP.


[Array partition with minimum difference](https://www.naukri.com/code360/problems/partition-a-set-into-two-subsets-such-that-the-difference-of-subset-sums-is-minimum._842494?utm_source=striver&utm_medium=website&utm_campaign=a_zcoursetuf)
```c++
int minSubsetSumDifference(vector<int>& nums, int n)
{
		int totSum = 0;
        for(int i=0; i<n; i++) totSum+=nums[i];

        int target = totSum;
        // //Now, Subset sum problem - Tabulation method;
        // vector<vector<bool>> dp(n, vector<bool>(target+1, 0));
        // for(int i=0; i<n; i++) dp[i][0]=true;
        // if(nums[0]<=target) dp[0][nums[0]]=true;
        // for(int i=1; i<n; i++) {
        //     for(int j=1; j<=target; j++) {
        //         bool not_take = dp[i-1][j];
        //         bool take = false;
        //         if(nums[i]<=j) {
        //             take = dp[i-1][j - nums[i]];
        //         }
        //         dp[i][j] = take||not_take;
        //     }
        // }
        // //Here, the dp[n-1][col->0...target] - will have T/F values indicating whether the targets(0...targets) are possible or not.
        // int mini = 1e9;
        // for(int i=0; i<=target; i++) {
        //     if(dp[n-1][i]==true) {
        //         mini = min(mini, abs((totSum-i)-i));
        //     }
        // }
        // return mini;

		//Now, Subset sum problem - Tabulation + Space Optimization method;
        vector<bool> prev(target+1, 0), cur(target+1, 0);
        for(int i=0; i<n; i++) prev[0]=true;
        if(nums[0]<=target) prev[nums[0]]=true;
        for(int i=1; i<n; i++) {
			cur[0] = true;
            for(int j=1; j<=target; j++) {
                bool not_take = prev[j];
                bool take = false;
                if(nums[i]<=j) {
                    take = prev[j - nums[i]];
                }
                cur[j] = take||not_take;
            }
			prev = cur;
        }
        //Here, the dp[n-1][col->0...target] - will have T/F values indicating whether the targets(0...targets) are possible or not.
        int mini = 1e9;
        for(int i=0; i<=target; i++) {
            if(prev[i]==true) {
                mini = min(mini, abs((totSum-i)-i));
            }
        }
        return mini;
}
```
**Key:** Use DP to find all achievable subset sums up to the total sum, then minimize the absolute difference between the two subset sums by iterating through possible sums.


[Perfect Sum Problem](https://www.geeksforgeeks.org/problems/perfect-sum-problem5633/1?utm_source=youtube&utm_medium=collab_striver_ytdescription&utm_campaign=perfect-sum-problem)
```c++
//{ Driver Code Starts
#include <bits/stdc++.h>
using namespace std;


// } Driver Code Ends
class Solution {
  public:
    // int fn(int idx, int target, vector<int>& arr, vector<vector<int>>& dp) {
    //     if(idx==0) {
    //         if(target==0 and arr[idx]==0) return 2;
    //         if(target==0 || target == arr[0]) return 1;
    //         return 0;
    //     }
    //     if(dp[idx][target]!=-1) return dp[idx][target];
    //     bool not_take = fn(idx-1, target, arr, dp);
    //     bool take = 0;
    //     if(arr[idx]<=target) {
    //         take = fn(idx-1, target-arr[idx], arr, dp);
    //     }
    //     return dp[idx][target] = take+not_take;
    // }
    int perfectSum(vector<int>& arr, int target) {
        int n = arr.size();
        //Memoizaiton: not working because the arr can contain 0. 
        //And, in that case, it can be both picked and not picked.
        // vector<vector<int>> dp(n, vector<int>(target+1, -1));
        // return fn(n-1, target, arr, dp);
        
        //Tabulation:
        vector<vector<int>> dp(n, vector<int>(target+1, 0));
        //for(int i=0; i<n; i++) dp[i][0]=1;
        
        if(arr[0]==0) dp[0][0]=2;
        else dp[0][0]=1;
        if(arr[0]!=0 and arr[0]<=target) dp[0][arr[0]] = 1;
        
        for(int i=1; i<n; i++) {
            for(int j=0; j<=target; j++) {
                int not_take = dp[i-1][j];
                int take = 0;
                if(arr[i]<=j) {
                    take = dp[i-1][j-arr[i]];
                }
                dp[i][j] = take+not_take;
            }
        }
        return dp[n-1][target];
    }
};
```
**Key:** Count all subsets with a given sum using DP by considering inclusion/exclusion of each element, accounting for zeros that can contribute multiple ways.


[Partitions with Given Difference](https://www.geeksforgeeks.org/problems/partitions-with-given-difference/1?utm_source=youtube&utm_medium=collab_striver_ytdescription&utm_campaign=partitions-with-given-difference)
```c++
//{ Driver Code Starts
// Initial function template for C++

#include <bits/stdc++.h>
using namespace std;


// } Driver Code Ends
class Solution {
  public:
    // int fn(int idx, int target, vector<int>& arr, vector<vector<int>>& dp) {
    //     if(idx==0) {
    //         if(target==0 and arr[0]==0) return 2;
    //         if(target==0 || arr[0]==target) return 1;
    //         else return 0;
    //     }
    //     if(dp[idx][target]!=-1) return dp[idx][target];
    //     int not_take = fn(idx-1, target, arr, dp);
    //     int take = false;
    //     if(arr[idx]<=target) {
    //         take = fn(idx-1, target-arr[idx], arr, dp);
    //     }
    //     return dp[idx][target] = take+not_take;
    // }
    // int subsetSum(int n, vector<int>& arr, int target) {
    //     vector<vector<int>> dp(n, vector<int>(target+1, -1));
    //     return fn(n-1, target, arr, dp);
    // }
    int countPartitions(vector<int>& arr, int d) {
        int totSum=0;
        for(int i=0; i<arr.size(); i++) totSum+=arr[i];
        if(totSum-d<0 || (totSum-d)%2) return 0;
        
        //Memoization:
        //return subsetSum(arr.size(), arr, (totSum-d)/2);
        int n = arr.size();
        int target = (totSum-d)/2;
        
        //Tabulation:
        // vector<vector<int>> dp(n, vector<int>(target+1, 0));
        // //Base cases
        // if(arr[0]==0) dp[0][0]=2;
        // else dp[0][0] = 1;
        // if(arr[0]!=0 and arr[0]<=target) dp[0][arr[0]]=1;
        
        // for(int i=1; i<n; i++) {
        //     for(int j=0; j<=target; j++) {
        //         int not_take = dp[i-1][j];
        //         int take = 0;
        //         if(arr[i]<=j) {
        //             take = dp[i-1][j-arr[i]];
        //         }
        //         dp[i][j] = take+not_take;
        //     }
        // }
        // return dp[n-1][target];
        
        //Tabulation + Space Optimization:
        vector<int> prev(target+1, 0), cur(target+1, 0);
        //Base cases
        if(arr[0]==0) prev[0]=2;
        else prev[0]=1;
        if(arr[0]!=0 and arr[0]<=target) prev[arr[0]]=1;
        
        for(int i=1; i<n; i++) {
            for(int j=0; j<=target; j++) {
                int not_take = prev[j];
                int take = 0;
                if(arr[i]<=j) {
                    take = prev[j-arr[i]];
                }
                cur[j] = take+not_take;
            }
            prev = cur;
        }
        return prev[target];
    }
};
```
**Key:** The problem is reduced to finding the number of subsets that sum up to a specific value, which is derived from the given difference. This transforms it into a classic subset sum problem


[Coin Change](https://leetcode.com/problems/coin-change/)
```c++
class Solution {
public:
    // long fn(int idx, int T, vector<int> &coins, vector<vector<int>> & dp) {
    //     if(idx==0) {
    //         if(T%coins[idx]==0) return T/coins[idx];
    //         return INT_MAX;
    //     }
    //     if(dp[idx][T]!=-1) return dp[idx][T];
    //     long not_take = 0 + fn(idx-1, T, coins, dp);
    //     long take = INT_MAX;
    //     if(coins[idx]<=T) {
    //         take = 1 + fn(idx, T-coins[idx], coins, dp);
    //     }
    //     return dp[idx][T] = min(take, not_take);
    // }
    int coinChange(vector<int>& coins, int amount) {
        int n = coins.size();
        //Memoization:
        // vector<vector<int>> dp(n, vector<int>(amount+1, -1));
        // long ans = fn(n-1, amount, coins, dp);
        
        //Tabulation:
        // vector<vector<int>> dp(n, vector<int>(amount+1, 0));
        // //Base case;
        // for(int T=0; T<=amount; T++) {
        //     if(T%coins[0]==0) dp[0][T] = T/coins[0];
        //     else dp[0][T] = 1e9;
        // }
        // //Building the steps
        // for(int i=1; i<n; i++) {
        //     for(int T=0; T<=amount; T++) {
        //         int not_take = 0 + dp[i-1][T];
        //         int take = INT_MAX;
        //         if(coins[i]<=T) {
        //             take = 1 + dp[i][T-coins[i]];
        //         }
        //         dp[i][T] = min(take, not_take);
        //     }
        // }
        // int ans = dp[n-1][amount];

        //Tabulation + Space Optimization:
        vector<int> prev(amount+1, 0), cur(amount+1, 0);
        //Base case;
        for(int T=0; T<=amount; T++) {
            if(T%coins[0]==0) prev[T] = T/coins[0];
            else prev[T] = 1e9;
        }
        //Building the steps
        for(int i=1; i<n; i++) {
            for(int T=0; T<=amount; T++) {
                int not_take = 0 + prev[T];
                int take = 1e9;
                if(coins[i]<=T) {
                    take = 1 + cur[T-coins[i]];
                }
                cur[T] = min(take, not_take);
            }
            prev = cur;
        }
        int ans = prev[amount];

        if(ans >= 1e9) return -1;
        return (int)ans;
    }
};
```
**Key:** The problem is about choosing coins to reach a target amount with the fewest coins. By considering each coin at every step, we either include it or exclude it, and use dynamic programming to build up the minimum number of coins required for each amount.


[Target Sum](https://leetcode.com/problems/target-sum/)
```c++
class Solution {
public:
    // int fn(int idx, int target, vector<int> &nums, vector<vector<int>> &dp) {
    //     if(idx==0) {
    //         if(target==0 and nums[0]==0) return 2;
    //         if(target==0 || target==nums[0]) return 1;
    //         else return 0;
    //     }
    //     if(dp[idx][target]!=-1) return dp[idx][target];
    //     int not_take = fn(idx-1, target, nums, dp);
    //     int take = 0;
    //     if(nums[idx]<=target) {
    //         take = fn(idx-1, target-nums[idx], nums, dp);
    //     }
    //     return dp[idx][target] = take+not_take;
    // }
    int subsetSum(int n, int d, vector<int> &nums) {
        int totsum = 0;
        for(int i=0; i<n; i++) totsum+=nums[i];
        if((totsum-d)<0 || (totsum-d)%2) return 0;
        int target = (totsum-d)/2;
        
        //Memoization:
        // vector<vector<int>> dp(n, vector<int>(target+1, -1));
        // return fn(n-1, target, nums, dp);
        
        //Tabulation:
        // vector<vector<int>> dp(n, vector<int>(target+1, 0));
        // //Base cases;
        // if(nums[0]==0) dp[0][0] = 2;
        // else dp[0][0] = 1;
        // if(nums[0]!=0 and nums[0]<=target) dp[0][nums[0]]=1;
        // //Bulding the states
        // for(int i=1; i<n; i++) {
        //     for(int j=0; j<=target; j++) {
        //         int not_take = dp[i-1][j];
        //         int take = 0;
        //         if(nums[i]<=j) {
        //             take = dp[i-1][j-nums[i]];
        //         }
        //         dp[i][j] = take+not_take;
        //     }
        // }
        // return dp[n-1][target];

        //Tabulation + Space Optimization:
        vector<int> prev(target+1, 0), cur(target+1, 0);
        //Base case;
        if(nums[0]==0) prev[0]=2;
        else prev[0]=1;
        if(nums[0]!=0 and nums[0]<=target) prev[nums[0]] = 1;
        for(int i=1; i<n; i++) {
            for(int j=0; j<=target; j++) {
                int not_take = prev[j];
                int take = 0;
                if(nums[i]<=j) {
                    take = prev[j-nums[i]];
                }
                cur[j] = take+not_take;
            }
            prev = cur;
        }
        return prev[target];
    }
    int findTargetSumWays(vector<int>& nums, int target) {
        int n = nums.size();
        int ans = subsetSum(n, target, nums);
        return ans;
    }
};
```
**Key:** The problem is reduced to finding the number of subsets that sum up to a specific value, which is derived from the given target sum. It’s solved by treating it as a subset sum problem and using dynamic programming to count the ways to achieve the target by either including or excluding each number.


[Coin Change II](https://leetcode.com/problems/coin-change-ii/)
```c++
class Solution {
public:
    // int fn(int idx, int target, vector<int> &coins, vector<vector<int>> &dp) {
    //     if(idx==0) {
    //         return (target%coins[idx]==0);
    //     }
    //     if(dp[idx][target]!=-1) return dp[idx][target];
    //     int not_take = fn(idx-1, target, coins, dp);
    //     int take = 0;
    //     if(coins[idx]<=target) {
    //         take = fn(idx, target - coins[idx], coins, dp);
    //     }
    //     return dp[idx][target] = take+not_take;
    // } 
    int change(int amount, vector<int>& coins) {
        int n = coins.size();

        //Memoization:
        // vector<vector<int>> dp(n, vector<int>(amount+1, -1));
        // return fn(n-1, amount, coins, dp);

        //Tabulation:
    //     vector<vector<double>> dp(n, vector<double>(amount+1, 0));
    //     //Base case;
    //     for(int i=0; i<=amount; i++) {
    //         if(i%coins[0]==0) dp[0][i] = 1; 
    //         else dp[0][i]=0;
    //     }
    //     //Building up the steps;
    //     for(int i=1; i<n; i++) {
    //         for(int j=0; j<=amount; j++) {
    //             double not_take = dp[i-1][j];
    //             double take = 0;
    //             if(coins[i]<=j) {
    //                 take = dp[i][j-coins[i]];
    //             }
    //             dp[i][j] = (take+not_take);
    //         }
    //     }
    //     return (int)dp[n-1][amount];
    

        //Tabulation + Space Optimization:
        vector<double> prev(amount+1, 0), cur(amount+1, 0);
        //Base case;
        for(int i=0; i<=amount; i++) {
            if(i%coins[0]==0) prev[i]=1;
            else prev[i]=0;
        }
        //Building up the steps;
        for(int i=1; i<n; i++) {
            for(int j=0; j<=amount; j++){
                double not_take = prev[j];
                double take = 0;
                if(coins[i]<=j) {
                    take = cur[j-coins[i]];
                }
                cur[j] = take+not_take;
            }
            prev=cur;
        }
        return (int)prev[amount];
    }
};
```
**Key:** Similar to above problems.


[Knapsack with Duplicate Items](https://www.geeksforgeeks.org/problems/knapsack-with-duplicate-items4201/1?utm_source=youtube&utm_medium=collab_striver_ytdescription&utm_campaign=knapsack-with-duplicate-items)
```c++
//{ Driver Code Starts
// Initial Template for C++

#include <bits/stdc++.h>
using namespace std;


// } Driver Code Ends
// User function Template for C++

class Solution {
  public:
    // int fn(int idx, int W, vector<int> &val, vector<int> &wt,
    // vector<vector<int>> &dp) {
    //     if(idx==0) {
    //         return ((int)(W/wt[0]))*val[0];
    //     }
    //     if(dp[idx][W]!=-1) return dp[idx][W];
    //     int not_take = 0 + fn(idx-1, W, val, wt, dp);
    //     int take = INT_MIN;
    //     if(wt[idx]<=W) {
    //         take = val[idx] + fn(idx, W-wt[idx], val, wt, dp);
    //     }
    //     return dp[idx][W] = max(take, not_take);
    // }
    int knapSack(vector<int>& val, vector<int>& wt, int W) {
        int n = val.size();
        
        //Memoization:
        // vector<vector<int>> dp(n, vector<int>(W+1, -1));
        // return fn(n-1, W, val, wt, dp);
        
        //Tabulation:
        // vector<vector<int>> dp(n, vector<int>(W+1, 0));
        // //Base case;
        // for(int w=0; w<=W; w++) {
        //     dp[0][w] = ((int)(w/wt[0]))*val[0];
        // }
        // //Building the steps;
        // for(int i=1; i<n; i++) {
        //     for(int j=0; j<=W; j++) {
        //         int not_take = 0 + dp[i-1][j];
        //         int take = INT_MIN;
        //         if(wt[i]<=j) {
        //             take = val[i] + dp[i][j-wt[i]];
        //         }
        //         dp[i][j] = max(take, not_take);
        //     }
        // }
        // return dp[n-1][W];
        
        //Tabulation + Space Optimization:
        // vector<int> prev(W+1, 0), cur(W+1, 0);
        // //Base case;
        // for(int w=0; w<=W; w++) {
        //     prev[w] = ((int)(w/wt[0]))*val[0];
        // }
        // //Building the steps;
        // for(int i=1; i<n; i++) {
        //     for(int j=0; j<=W; j++) {
        //         int not_take = 0 + prev[j];
        //         int take = INT_MIN;
        //         if(wt[i]<=j) {
        //             take = val[i] + cur[j-wt[i]];
        //         }
        //         cur[j] = max(take, not_take);
        //     }
        //     prev = cur;
        // }
        // return prev[W];
        
        //Tabulation + 1D Space Optimization:
        vector<int> prev(W+1, 0);
        //Base case;
        for(int w=0; w<=W; w++) {
            prev[w] = ((int)(w/wt[0]))*val[0];
        }
        //Building the steps;
        for(int i=1; i<n; i++) {
            for(int j=0; j<=W; j++){
                int not_take = 0 + prev[j];
                int take = INT_MIN;
                if(wt[i]<=j) {
                    take = val[i] + prev[j-wt[i]];
                }
                prev[j] = max(take, not_take);
            }
        }
        return prev[W];  
    }
};
```
**Key:** The key idea is to maximize value by either excluding the current item (carry forward the previous value) or including it (add its value and stay at the same index with reduced capacity), allowing multiple picks of the same item.


[Rod Cutting](https://www.geeksforgeeks.org/problems/rod-cutting0840/1)
```c++
//{ Driver Code Starts
// Initial Template for C++

#include <bits/stdc++.h>
using namespace std;


// } Driver Code Ends
// User function Template for C++

class Solution {
  public:
    int fn(int idx, int N, vector<int> & price, vector<vector<int>> &dp) {
        if(idx==0) {
            return N*price[0];
        }
        if(dp[idx][N]!=-1) return dp[idx][N];
        int not_take = 0 + fn(idx-1, N, price, dp);
        int take = INT_MIN;
        int rodLength = idx+1;
        if(rodLength<=N) {
            take = price[idx] + fn(idx, N - rodLength, price, dp);
        }
        return dp[idx][N] = max(take, not_take);
    }
    int cutRod(vector<int> &price) {
        int n = price.size();
        
        //Memoization:
        // vector<vector<int>> dp(n, vector<int>(n+1, -1));
        // return fn(n-1, n, price, dp);
        
        //Tabulation:
        // vector<vector<int>> dp(n, vector<int>(n+1, 0));
        // //Base case;
        // for(int i=0; i<=n; i++) {
        //     dp[0][i] = i*price[0];
        // }
        // //Building the steps;
        // for(int i=1; i<n; i++) {
        //     for(int j=0; j<=n; j++) {
        //         int not_take = 0 + dp[i-1][j];
        //         int take = INT_MIN;
        //         int rodLength = i+1;
        //         if(rodLength<=j) {
        //             take = price[i] + dp[i][j-rodLength];
        //         }
        //         dp[i][j] = max(take, not_take);
        //     }
        // }
        // return dp[n-1][n];
        
        //Tabulation + Space Optimization:
        // vector<int> prev(n+1, 0), cur(n+1, 0);
        // //Base case;
        // for(int i=0; i<=n; i++) {
        //     prev[i] = i*price[0];
        // }
        // //Building the steps;
        // for(int i=1; i<n; i++) {
        //     for(int j=0; j<=n; j++) {
        //         int not_take = 0 + prev[j];
        //         int take = INT_MIN;
        //         int rodLength = i+1;
        //         if(rodLength<=j) {
        //             take = price[i] + cur[j-rodLength];
        //         }
        //         cur[j] = max(take, not_take);
        //     }
        //     prev = cur;
        // }
        // return prev[n];
        
        //Tabulation + 1D Space Optimization:
        vector<int> prev(n+1, 0);
        //Base case;
        for(int i=0; i<=n; i++) {
            prev[i] = i*price[0];
        }
        //Building the steps;
        for(int i=1; i<n; i++) {
            for(int j=0; j<=n; j++) {
                int not_take = 0 + prev[j];
                int take = INT_MIN;
                int rodLength = i+1;
                if(rodLength<=j) {
                    take = price[i] + prev[j-rodLength];
                }
                prev[j] = max(take, not_take);
            }
        }
        return prev[n];
    }
};
```
**Key:** We can approach this differently by aiming to select rod lengths that sum up to N, maximizing the total price by considering both the options of picking or not picking each rod length.


---

Let’s conclude here, and I’ll catch you in the next one!