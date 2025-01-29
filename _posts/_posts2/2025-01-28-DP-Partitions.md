---
layout: post
title: MCM DP | DP on Partitions (Patterns and Problems)
date: 2025-01-25 20:07:00-0400
featured: false
description: A structured reference set of DP on MCM/Partitions problems, directly adapted from Striver's DP playlist.
tags: DP Code
categories: DSA
giscus_comments: false
related_posts: false
# toc:
#   sidebar: left
---

#### **Problems:**

[Matrix Chain Multiplication](https://www.geeksforgeeks.org/problems/matrix-chain-multiplication0303/1?utm_source=youtube&utm_medium=collab_striver_ytdescription&utm_campaign=matrix-chain-multiplication)
```c++
class Solution {
  public:
    // int fn(int i, int j, vector<int> & arr, vector<vector<int>> &dp) {
    //     //Base case;
    //     if(i==j) return 0;
    //     //Partitions;
    //     int mini = 1e9;
    //     int steps=0;
    //     if(dp[i][j]!=-1) return dp[i][j];
    //     for(int k=i; k<=j-1; k++) {
    //         steps = arr[i-1]*arr[k]*arr[j] + fn(i, k, arr, dp)
    //             + fn(k+1, j, arr, dp);
    //         mini = min(mini, steps);
    //     }
    //     return dp[i][j] = mini;
    // }
    int matrixMultiplication(vector<int> &arr) {
        int n = arr.size();
        //Memoization:
        // vector<vector<int>> dp(n, vector<int>(n, -1));
        // return fn(1, n-1, arr, dp);
        
        //Tabulation:
        vector<vector<int>> dp(n, vector<int>(n, 0));
        //Base case: not required as its correctly populated with default 
        //value, but again mentioning it for clarity of conversion from
        //memoization to tabulation;
        for(int i=0; i<n; i++) {
            dp[i][i] = 0;
        }
        //Recurrence;
        for(int i=n-1; i>=0; i--) {
            for(int j=i+1; j<n; j++) {
                int mini = 1e9;
                int steps = 0;
                for(int k=i; k<=j-1; k++) {
                    steps = arr[i-1]*arr[k]*arr[j] + dp[i][k] + dp[k+1][j];
                    mini = min(mini, steps);
                }
                dp[i][j] = mini;
            }
        }
        return dp[1][n-1];
        
        //Tabulation + Space Optimization: Not Applicable, as we don't have
        //uniformity in index calculations of the dp table;
    }
};
```
**Key:** Refer to the [Link1](https://www.youtube.com/watch?v=vRVfmbCFW7Y) and [Link2](https://www.youtube.com/watch?v=pDCXsbAw5Cg)


[Minimum Cost to Cut a Stick](https://leetcode.com/problems/minimum-cost-to-cut-a-stick/)
```c++
class Solution {
public:
    // int fn(int i, int j, vector<int> &cuts,
    //     vector<vector<int>> &dp) {
    //     //Base case;
    //     if(i>j) return 0;
    //     //Recurrence;
    //     if(dp[i][j]!=-1) return dp[i][j];
    //     int mini = 1e9;
    //     int cost = 0;
    //     for(int ind=i; ind<=j; ind++) {
    //         cost = (cuts[j+1] - cuts[i-1]) + fn(i, ind-1, cuts, dp)
    //             + fn(ind+1, j, cuts, dp);
    //         mini = min(mini, cost);
    //     }
    //     return dp[i][j] = mini;
    // }
    int minCost(int n, vector<int>& cuts) {
        int c = cuts.size();
        //Adding 0 and n(stick length) for easier calculation;
        cuts.push_back(n);
        cuts.insert(cuts.begin(), 0);
        sort(cuts.begin(), cuts.end());

        //Memoization:
        // vector<vector<int>> dp(c+1, vector<int>(c+1, -1));
        // return fn(1, c, cuts, dp);

        //Tabulation:
        vector<vector<int>> dp(c+2, vector<int>(c+2, 0));
        //Base case: already populated with default values 0;
        //Recurrence;
        for(int i=c; i>=1; i--) {
            for(int j=i; j<=c; j++) {
                //if(i>j) continue;
                int mini = 1e9;
                int cost = 0;
                for(int ind=i; ind<=j; ind++) {
                    cost = (cuts[j+1] - cuts[i-1]) + dp[i][ind-1]
                        + dp[ind+1][j];
                    mini = min(mini, cost);
                }
                dp[i][j] = mini;
            }
        }
        return dp[1][c];
    }
};
```
**Key:** Refer to the [Link]()


[Burst Balloons](https://leetcode.com/problems/burst-balloons/)
```c++
class Solution {
public:
    // int fn(int i, int j, vector<int> &nums, vector<vector<int>> &dp) {
    //     //Base case;
    //     if(i>j) return 0;
    //     //Recurrence;
    //     if(dp[i][j]!=-1) return dp[i][j];
    //     int maxi = INT_MIN;
    //     int cost = 0;
    //     for(int ind=i; ind<=j; ind++) {
    //         cost = nums[i-1]*nums[ind]*nums[j+1] + fn(i, ind-1, nums, dp) 
    //             + fn(ind+1, j, nums, dp);
    //         maxi = max(cost, maxi);
    //     }
    //     return dp[i][j] = maxi;
    // }
    int maxCoins(vector<int>& nums) {
        int n = nums.size();
        nums.push_back(1);
        nums.insert(nums.begin(), 1);

        //Memoization:
        // vector<vector<int>> dp(n+1, vector<int>(n+1, -1));
        // return fn(1, n, nums, dp);

        //Tabulation:
        vector<vector<int>> dp(n+2, vector<int>(n+1, 0));
        //Base case: already covered with default values 0, Plus a small
        //addition will the made in the looping;
        //Recurrence;
        for(int i=n; i>=1; i--) {
            for(int j=1; j<=n; j++) {
                //Base case;
                if(i>j) continue;
                int cost = 0;
                int maxi = INT_MIN;
                for(int ind=i; ind<=j; ind++) {
                    cost = nums[i-1]*nums[ind]*nums[j+1] + dp[i][ind-1]
                        + dp[ind+1][j];
                    maxi = max(cost, maxi);
                }
                dp[i][j] = maxi;
            }
        }
        return dp[1][n];
    }
};
```
**Key:** Refer to the [Link]()


[Boolean Evaluation](https://www.naukri.com/code360/problems/boolean-evaluation_1214650?utm_source=striver&utm_medium=website&utm_campaign=a_zcoursetuf)
```c++
#define ll long long
int mod = 1000000007;
// long long fn(int i, int j, int isTrue, string &exp, 
//     vector<vector<vector<ll>>> &dp) {
//     //Base cases;
//     if(i>j) return 0; //no partition
//     if(i==j) {
//         if(isTrue) return exp[i]=='T';
//         else return exp[i]=='F';
//     }
//     //Recurrence;
//     if(dp[i][j][isTrue]!=-1) return dp[i][j][isTrue];
//     ll ways = 0;
//     //do partitions on the operator positions, which is from
//     //i+1 to j-1 on increments of two;
//     for(int ind=i+1; ind<=j-1; ind=ind+2) {
//         ll lT = fn(i, ind-1, 1, exp, dp);
//         ll lF = fn(i, ind-1, 0, exp, dp);
//         ll rT = fn(ind+1, j, 1, exp, dp);
//         ll rF = fn(ind+1, j, 0, exp, dp);

//         //calculating ways based on the operator;
//         if(exp[ind]=='&') {
//             if(isTrue) {
//                 ways = (ways + (lT*rT)%mod)%mod;
//             }
//             else {
//                 ways = (ways + (lT*rF)%mod + (lF*rT)%mod + (lF*rF)%mod 
//                 )%mod;
//             }
//         }
//         else if(exp[ind]=='|') {
//             if(isTrue) {
//                 ways = (ways + (lT*rT)%mod + (lT*rF)%mod + (lF*rT)%mod
//                 )%mod;
//             }
//             else {
//                 ways = (ways + (lF*rF)%mod)%mod;
//             }
//         }
//         else {
//             if(isTrue) {
//                 ways = (ways + (lT*rF)%mod + (lF*rT)%mod
//                 )%mod;
//             }
//             else {
//                 ways = (ways + (lF*rF)%mod + (lT*rT)%mod
//                 )%mod;
//             }
//         }
//     }
//     return dp[i][j][isTrue] = ways%mod;
// }
int evaluateExp(string & exp) {
    int n = exp.size();

    //Memoization:
    //vector<vector<vector<ll>>> dp(n, vector<vector<ll>>(n, vector<ll>(2, -1)));
    //return (int)fn(0, n-1, 1, exp, dp);

    //Tabulation:
    vector<vector<vector<ll>>> dp(n+1, vector<vector<ll>>(n+1, vector<ll>(2, 0)));
    //Base case;
    for(int i=0; i<n; i++) {
        
    }
    //Recurrence;
    for(int i=n-1; i>=0; i--) {
        for(int j=0; j<=n-1; j++) {
            if (i>j) continue;
            else if(i==j) {
                dp[i][i][1] = (exp[i]=='T');
                dp[i][i][0] = (exp[i]=='F');
            }
            else {
                ll waysT = 0;
                ll waysF = 0;
                //do partitions on the operator positions, which is from
                //i+1 to j-1 on increments of two;
                for(int ind=i+1; ind<=j-1; ind=ind+2) {
                    ll lT = dp[i][ind-1][1]; 
                    ll lF = dp[i][ind-1][0]; 
                    ll rT = dp[ind+1][j][1]; 
                    ll rF = dp[ind+1][j][0]; 

                    //calculating ways based on the operator;
                    if(exp[ind]=='&') {
                        waysT = (waysT + (lT*rT)%mod)%mod;

                        waysF = (waysF + (lT*rF)%mod + (lF*rT)%mod + (lF*rF)%mod 
                        )%mod;
                    }
                    else if(exp[ind]=='|') {
                        waysT = (waysT + (lT*rT)%mod + (lT*rF)%mod + (lF*rT)%mod
                        )%mod;


                        waysF = (waysF + (lF*rF)%mod)%mod;
                    } else {
                        waysT = (waysT + (lT * rF) % mod + (lF * rT) % mod) % mod;

                        waysF = (waysF + (lF * rF) % mod + (lT * rT) % mod) % mod;
                    }
                }
                dp[i][j][1] = waysT % mod;
                dp[i][j][0] = waysF % mod;
            }
        }
    }
    return dp[0][n-1][1];
}
```
**Key:** Refer to the [Link]()


[Palindrome Partitioning II](https://leetcode.com/problems/palindrome-partitioning-ii/)
```c++
class Solution {
public:
    int isPalindrome(int i, int j, string &s) {
        while(i<j) {
            if(s[i]!=s[j]) return false;
            i++;
            j--;
        }
        return true;
    }
    // int fn(int ind, int n, string &s, vector<int> &dp) {
    //     //Base case;
    //     if(ind==n) return 0;
    //     //Recurrence;
    //     if(dp[ind]!=-1) return dp[ind];
    //     int mini = INT_MAX;
    //     int cost = 0;
    //     for(int j = ind; j<n; j++) {
    //         if(isPalindrome(ind, j, s)) {
    //             cost = 1 + fn(j+1, n, s, dp);
    //             mini = min(mini, cost);
    //         }
    //     }
    //     return dp[ind] = mini;
    // }
    int minCut(string s) {
        //Front Partition Approach;
        int n = s.size();
        //Memoization:
        // vector<int> dp(n, -1);
        // return fn(0, n, s, dp)-1;      

        //Tabulation:
        vector<int> dp(n+1, 0);
        //Base case:
        dp[n] = 0;
        //Recurrence;
        for(int ind=n-1; ind>=0; ind--) {
            int mini = INT_MAX;
            int cost = 0;
            for(int j = ind; j<n; j++) {
                if(isPalindrome(ind, j, s)) {
                    cost = 1 + dp[j+1];
                    mini = min(mini, cost);
                }
            }
            dp[ind] = mini;
        }
        return dp[0]-1;  
    }
};
```
**Key:** Refer to the [Link]()


[Partition Array for Maximum Sum](https://leetcode.com/problems/partition-array-for-maximum-sum/)
```c++
class Solution {
public:
    // int fn(int ind, int n, vector<int> &arr, int k, vector<int> &dp) {
    //     //Base case;
    //     if(ind==n) return 0;
    //     //Recurrence;
    //     if(dp[ind]!=-1) return dp[ind];
    //     int maxElement = INT_MIN;
    //     int cost = 0;
    //     int len = 0;
    //     int maxAns = INT_MIN;
    //     for(int j = ind; j<min(ind+k, n); j++) {
    //         len++;
    //         maxElement = max(maxElement, arr[j]);
    //         cost = len*maxElement + fn(j+1, n, arr, k, dp);
    //         maxAns = max(cost, maxAns);
    //     }
    //     return dp[ind] = maxAns;
    // }
    int maxSumAfterPartitioning(vector<int>& arr, int k) {
        //Front Partition Approach;
        int n = arr.size();
        //Memoization:
        // vector<int> dp(n, -1);
        // return fn(0, n, arr, k, dp);

        //Tabulation:
        vector<int> dp(n+1, 0);
        //Base case;
        dp[n] = 0;
        //Recurrence;
        for(int ind=n-1; ind>=0; ind--) {
            int maxElement = INT_MIN;
            int cost = 0;
            int len = 0;
            int maxAns = 0;
            for(int j = ind; j<min(ind+k, n); j++) {
                len++;
                maxElement = max(maxElement, arr[j]);
                cost = len*maxElement + dp[j+1];
                maxAns = max(maxAns, cost);
            }
            dp[ind] = maxAns;
        }
        return dp[0];        
    }
};
```
**Key:** Refer to the [Link]()

---

Let’s conclude here, and I’ll catch you in the next one!