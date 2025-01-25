---
layout: post
title: DP on Strings (Patterns and Problems)
date: 2025-01-25 00:27:00-0400
featured: false
description: A structured reference set of DP on Strings problems, directly adapted from Striver's DP playlist.
tags: DP Code
categories: DSA
giscus_comments: false
related_posts: false
# toc:
#   sidebar: left
---

[Longest Common Subsequence](https://leetcode.com/problems/longest-common-subsequence/)
```c++
class Solution {
public:
    // int fn(int i, int j, string &s1, string &s2, vector<vector<int>> &dp) {
    //     if(i<0 || j<0) return 0;
    //     if(dp[i][j]!=-1) return dp[i][j];
    //     if(s1[i]==s2[j]) return dp[i][j] =  1 + fn(i-1, j-1, s1, s2, dp);
    //     return dp[i][j] = 0 + max(fn(i-1, j, s1, s2, dp), fn(i, j-1, s1, s2, dp));
    // }
    int longestCommonSubsequence(string text1, string text2) {
        int n = text1.size();
        int m = text2.size();

        //Memoization:
        // vector<vector<int>> dp(n, vector<int>(m, -1));
        // return fn(n-1, m-1, text1, text2, dp);

        //Tabulation:
        // vector<vector<int>> dp(n+1, vector<int>(m+1, 0));
        // //Base case;
        // for(int i=0; i<=n; i++) dp[i][0] = 0;
        // for(int j=0; j<=m; j++) dp[0][j] = 0;
        // //Building the steps;
        // for(int i=1; i<=n; i++) {
        //     for(int j=1; j<=m; j++) {
        //         if(text1[i-1]==text2[j-1]) dp[i][j] = 1 + dp[i-1][j-1];
        //         else dp[i][j] = 0 + max(dp[i-1][j], dp[i][j-1]);
        //     }
        // }
        // return dp[n][m];

        //Tabulation + Space Optimization:
        vector<int> prev(m+1, 0), cur(m+1, 0);
        //Base case;
        for(int i=0; i<=m; i++) prev[i] = 0;
        //Building the steps;
        for(int i=1; i<=n; i++) {
            for(int j=1; j<=m; j++) {
                if(text1[i-1]==text2[j-1]) cur[j] = 1 + prev[j-1];
                else cur[j] = 0 + max(cur[j-1], prev[j]);
            }
            prev = cur;
        }
        return prev[m];
    }
};
```
**Key:** Use a 2D DP array where dp[i][j] stores the length of the LCS of the first i characters of text1 and the first j characters of text2. If the characters match, include it in the LCS; otherwise, take the maximum LCS excluding one character at a time from either of the texts.


[Print all LCS sequences](https://www.geeksforgeeks.org/problems/print-all-lcs-sequences3413/1?utm_source=youtube&utm_medium=collab_striver_ytdescription&utm_campaign=print-all-lcs-sequences)
```c++
class Solution {
  public:
    vector<string> all_longest_common_subsequences(string s, string t) {
        //Tabulation method to find the LCS(Count):
        int n = s.size();
        int m = t.size();
        vector<vector<int>> dp(n+1, vector<int>(m+1, 0));
        //Base case;
        for(int i=0; i<=n; i++) dp[i][0] = 0;
        for(int j=0; j<=m; j++) dp[0][j] = 0;
        //Building the steps;
        for(int i=1; i<=n; i++) {
            for(int j=1; j<=m; j++) {
                if(s[i-1]==t[j-1]) dp[i][j] = 1 + dp[i-1][j-1];
                else dp[i][j] = 0 + max(dp[i-1][j], dp[i][j-1]);
            }
        }
        int lcs_len = dp[n][m];
        //cout<<lcs_len<<endl;
        //Print all the sub-sequence thats present with lcs calculation;
        int i = n-1;
        int j = m-1;
        vector<string> ans;
        string tmp = "X";
        for(int i=1; i<lcs_len; i++) tmp+='X';
        while(i>=0 and j>=0) {
            if(s[i]==t[j]) {
                tmp[lcs_len-1] = s[i];
                i--;
                j--;
                lcs_len--;
            }
            else if(dp[i-1][j]>dp[i][j]) {
                i--;
            }
            else j--;
        }
        //cout<<tmp<<endl;
        ans.push_back(tmp);
        return ans;
    }
};
```
**Key:** Use a 2D DP table to calculate the length of the LCS. Then, backtrack from the bottom-right corner of the table to collect the subsequence. During backtracking, if the characters match, include the character in the subsequence and decrement both the row and column indices. If the characters do not match, move to the direction with the higher value, either from the (i-1)th row or the (j-1)th column, and continue backtracking from that cell.


[Longest Common Substring](https://www.geeksforgeeks.org/problems/longest-common-substring1452/1)
```c++
class Solution {
  public:
    int longestCommonSubstr(string& s1, string& s2) {
        int n = s1.size();
        int m = s2.size();
        int ans = 0;
        //Tabulation:
        // vector<vector<int>> dp(n+1, vector<int>(m+1, 0));
        // //Base case;
        // for(int i=0; i<=n; i++) dp[i][0] = 0;
        // for(int j=0; j<=m; j++) dp[0][j] = 0;
        // //building the steps;
        // for(int i=1; i<=n; i++) {
        //     for(int j=1; j<=m; j++) {
        //         if(s1[i-1]==s2[j-1]) {
        //             dp[i][j] = 1 + dp[i-1][j-1];
        //             ans = max(ans, dp[i][j]);
        //         }
        //         else dp[i][j] = 0;
        //     }
        // }
        // return ans;
        
        //Tabulation + Space Optimization:
        vector<int> prev(m+1, 0), cur(m+1, 0);
        for(int i=1; i<=n; i++) {
            for(int j=1; j<=m; j++) {
                if(s1[i-1]==s2[j-1]) {
                    cur[j] = 1 + prev[j-1];
                    ans = max(ans, cur[j]);
                }
                else cur[j] = 0;
            }
            prev = cur;
        }
        return ans;
    }
};
```
**Key:** Use a 2D DP table where dp[i][j] stores the length of the longest common substring ending at indices i-1 and j-1 of s1 and s2. Update the result whenever a match is found, and reset to 0 if characters mismatch.


[Longest Palindromic Subsequence](https://leetcode.com/problems/longest-palindromic-subsequence/)
```c++
class Solution {
public:
    int fn(int i, int j, string s, string t, vector<vector<int>> & dp) {
        if(i<0 || j<0) return 0;
        if(dp[i][j]!=-1) return dp[i][j];
        if(s[i]==t[j]) return dp[i][j] = 1 + fn(i-1, j-1, s, t, dp);
        return dp[i][j] = 0 + max(fn(i-1, j, s, t, dp), fn(i, j-1, s, t, dp));
    }
    int longestPalindromeSubseq(string s) {
        string t = s;
        reverse(t.begin(), t.end());
        int n = s.size();

        //Memoization: MLE - Memory Limit Exceeded
        // vector<vector<int>> dp(n, vector<int>(n, -1));
        // return fn(n-1, n-1, s, t, dp);

        //Tabulation:
        vector<vector<int>> dp(n+1, vector<int>(n+1, 0));
        for(int i=1; i<=n; i++) {
            for(int j=1; j<=n; j++) {
                if(s[i-1]==t[j-1]) dp[i][j] = 1 + dp[i-1][j-1];
                else dp[i][j] = 0 + max(dp[i][j-1], dp[i-1][j]);
            }
        }
        return dp[n][n];

        //Tabulation  + Space Optimization:
        vector<int> prev(n+1, 0), cur(n+1, 0);
        for(int i=1; i<=n; i++) {
            for(int j=1; j<=n; j++) {
                if(s[i-1]==t[j-1]) cur[j] = 1 + prev[j-1];
                else cur[j] = 0 + max(prev[j], prev[j-1]);
            }
            prev = cur;
        }
        return prev[n];
    }
};
```
**Key:** Find the Longest Common Subsequence (LCS) between the string and its reverse to identify the longest palindromic subsequence.


[Minimum Insertion Steps to Make a String Palindrome](https://leetcode.com/problems/minimum-insertion-steps-to-make-a-string-palindrome/)
```c++
class Solution {
public:
    int lcs(string &s, string &t) {
        int n = s.size();
        int m = t.size();

        //Tabulation:
        // vector<vector<int>> dp(n+1, vector<int>(m+1, 0));
        // for(int i=1; i<=n; i++) {
        //     for(int j=1; j<=m; j++) {
        //         if(s[i-1]==t[j-1]) dp[i][j] = 1 + dp[i-1][j-1];
        //         else dp[i][j] = 0 + max(dp[i-1][j], dp[i][j-1]);
        //     }
        // }
        // return dp[n][m];

        //Tabulation + Space Optmization:
        vector<int> prev(m+1, 0), cur(m+1, 0);
        for(int i=1; i<=n; i++) {
            for(int j=1; j<=m; j++) {
                if(s[i-1]==t[j-1]) cur[j] = 1 + prev[j-1];
                else cur[j] = 0 + max(cur[j-1], prev[j]);
            }
            prev = cur;
        }
        return prev[m];
    }
    int minInsertions(string s) {
        //Via Longest Palindromic Subsequence;
        string t = s;
        reverse(t.begin(), t.end());
        return (s.size() - lcs(s, t));
        
    }
};
```
**Key:** Find the Longest Palindromic Subsequence (LPS) using LCS between the string and its reverse; the minimum insertions required are the difference between the string's length and its LPS.


[Delete Operation for Two Strings](https://leetcode.com/problems/delete-operation-for-two-strings/)
```c++
class Solution {
public:
    int lcs(string &s, string &t) {
        int n = s.size();
        int m = t.size();

        //Tabulation:
        // vector<vector<int>> dp(n+1, vector<int>(m+1, 0));
        // for(int i=1; i<=n; i++) {
        //     for(int j=1; j<=m; j++) {
        //         if(s[i-1]==t[j-1]) dp[i][j] = 1 + dp[i-1][j-1];
        //         else dp[i][j] = 0 + max(dp[i][j-1], dp[i-1][j]);
        //     }
        // }
        // return dp[n][m];

        //Tabulation + Space Optimization:
        vector<int> prev(m+1, 0), cur(m+1, 0);
        for(int i=1; i<=n; i++) {
            for(int j=1; j<=m; j++) {
                if(s[i-1]==t[j-1]) cur[j] = 1 + prev[j-1];
                else cur[j] = 0 + max(cur[j-1], prev[j]);
            }
            prev = cur;
        }
        return prev[m];
    }
    int minDistance(string word1, string word2) {
        //Via LCS Approach;
        return (word1.size() + word2.size() - 2*lcs(word1, word2));
    }
};
```
**Key:** Calculate the Longest Common Subsequence (LCS) of the two strings; the minimum deletions required are the total length of both strings minus twice the LCS length.


[Shortest Common Supersequence](https://leetcode.com/problems/shortest-common-supersequence/)
```c++
class Solution {
public:
    string shortestCommonSupersequence(string s, string t) {
        //via LCS approach;
        int n = s.size();
        int m = t.size();
        //Tabulation: Need to build the DP table to identify this string. If it's the size of the shortest common supersequence, then we can use the tabulation with space optmization(without the dp table);
        vector<vector<int>> dp(n+1, vector<int>(m+1, 0));
        for(int i=1; i<=n; i++) {
            for(int j=1; j<=m; j++) {
                if(s[i-1]==t[j-1]) dp[i][j] = 1 + dp[i-1][j-1];
                else dp[i][j] = 0 + max(dp[i-1][j], dp[i][j-1]);
            }
        }
        // dp[n][m] has the LCS size;
        int i = n;
        int j = m;
        string ans = "";
        while(i>0 and j>0) {
            if(s[i-1]==t[j-1]) {
                ans+=s[i-1];
                i--;
                j--;
            }
            else if(dp[i-1][j] > dp[i][j-1]) {
                ans+=s[i-1];
                i--;
            }
            else {
                ans+=t[j-1];
                j--;
            }
        }
        while(i>0) {ans+=s[i-1]; i--;}
        while(j>0) {ans+=t[j-1]; j--;}
        reverse(ans.begin(), ans.end());
        return ans;
    }
};
```
**Key:** Use the Longest Common Subsequence (LCS) approach to construct the Shortest Common Supersequence (SCS). First, build the DP table for the LCS. Then, use the table to form the SCS by including the LCS characters once and appending the remaining characters from both strings, ensuring the order is preserved. Finally, reverse the string formed during the backtracking process from the bottom-right of the DP table to obtain the correct SCS.


[Distinct Subsequences](https://leetcode.com/problems/distinct-subsequences/)
```c++
class Solution {
public:
    // int fn(int i, int j, string &s, string &t, vector<vector<double>> &dp) {
    //     if(j==0) return 1;
    //     if(i==0) return 0;
    //     if(dp[i][j]!=-1) return dp[i][j];
    //     if(s[i-1]==t[j-1]) {
    //         return dp[i][j] = fn(i-1, j-1, s, t, dp) + fn(i-1, j, s, t, dp);
    //     }
    //     return dp[i][j] = fn(i-1, j, s, t, dp);
    // }
    int numDistinct(string s, string t) {
        int n = s.size();
        int m = t.size();

        //Memoization:
        // vector<vector<double>> dp(n+1, vector<double>(m+1, -1));
        // return (int)fn(n, m, s, t, dp);

        //Tabulation:
        // vector<vector<double>> dp(n+1, vector<double>(m+1, 0));
        // //Base case;
        // for(int i=0; i<=n; i++) dp[i][0] = 1;
        // for(int j=1; j<=m; j++) dp[0][j] = 0;
        // //Building the steps;
        // for(int i=1; i<=n; i++) {
        //     for(int j=1; j<=m; j++) {
        //         if(s[i-1]==t[j-1]) {
        //             dp[i][j] = dp[i-1][j-1] + dp[i-1][j];
        //         }
        //         else dp[i][j] = dp[i-1][j];
        //     }
        // }
        // return dp[n][m];

        //Tabulation + Space Optimization:
        // vector<double> prev(m+1, 0), cur(m+1, 0);
        // prev[0] = cur[0] = 1;
        // for(int i=1; i<=n; i++) {
        //     for(int j=1; j<=m; j++) {
        //         if(s[i-1]==t[j-1]) {
        //             cur[j] = prev[j-1] + prev[j];
        //         }
        //         else {
        //             cur[j] = prev[j];
        //         }
        //     }
        //     prev = cur;
        // }
        // return (int)prev[m];

        //Tabulation + 1D Array Space Optimization:
        vector<double> prev(m+1, 0);
        prev[0] = 1;
        for(int i=1; i<=n; i++) {
            //Iterate from back: This is needed
            for(int j=m; j>=1; j--) {
                if(s[i-1]==t[j-1]) {
                    prev[j] = prev[j-1] + prev[j];
                }
            }
        }
        return (int)prev[m];
    }
};
```
**Key:** Check the recurrence relation; it will help you understand the approach.


[Edit Distance](https://leetcode.com/problems/edit-distance/)
```c++
class Solution {
public:
    // int fn(int i, int j, string &s, string &t,
    // vector<vector<int>> &dp) {
    //     //Base case;
    //     if(i<0) return j+1; //kind of insert to form s2
    //     if(j<0) return i+1; //kind of delete to form s2
    //     if(dp[i][j]!=-1) return dp[i][j];
    //     //Recurrence;
    //     if(s[i]==t[j]) return dp[i][j] = 0 + fn(i-1, j-1, s, t, dp);
    //     return dp[i][j] =  1 + min(
    //         fn(i, j-1, s, t, dp), //insert;
    //         min(fn(i-1, j, s, t, dp), //delete;
    //         fn(i-1, j-1, s, t, dp) //replace;
    //         )
    //     );

    // }
    int minDistance(string s, string t) {
        int n = s.size();
        int m = t.size();

        //Memoization:
        // vector<vector<int>> dp(n, vector<int>(m, -1));
        // return fn(n-1, m-1, s, t, dp);

        //Tabulation:
        // vector<vector<int>> dp(n+1, vector<int>(m+1, 0));
        // //Base case;
        // for(int i=0; i<=n; i++) dp[i][0] = i;
        // for(int j=0; j<=m; j++) dp[0][j] = j;
        // //Building the steps;
        // for(int i=1; i<=n; i++) {
        //     for(int j=1; j<=m; j++) {
        //         if(s[i-1]==t[j-1]) {
        //             dp[i][j] = 0 + dp[i-1][j-1];
        //         }
        //         else {
        //             dp[i][j] = 1 + min(
        //                 dp[i][j-1], //insert;
        //                 min(
        //                     dp[i-1][j], //delete;
        //                     dp[i-1][j-1] //replace;
        //                 )
        //             );
        //         }
        //     }
        // }
        // return dp[n][m];

        //Tabulation + Space Optmization:
        vector<int> prev(m+1, 0), cur(m+1, 0);
        //Base case;
        for(int j=0; j<=m; j++) prev[j] = j;
        //Building the steps;
        for(int i=1; i<=n; i++) {
            cur[0] = i;
            for(int j=1; j<=m; j++) {
                if(s[i-1]==t[j-1]) {
                    cur[j] = 0 + prev[j-1];
                }
                else {
                    cur[j] = 1 + min(
                        cur[j-1], //insert
                        min(
                            prev[j], //delete
                            prev[j-1]
                        )
                    );
                }
            }
            prev = cur;
        }
        return prev[m];
    }
};
```
**Key:** Check the recurrence relation; it will help you understand the approach.


[Wildcard Matching](https://leetcode.com/problems/wildcard-matching/)
```c++
class Solution {
public:
    // bool fn(int i, int j, string &p, string &s, vector<vector<int>> &dp) {
    //     //Base cases;
    //     if(i<0 && j<0) return 1;
    //     if(i<0 && j>=0) return 0;
    //     if(j<0 && i>=0) {
    //         //This case is due to * can take empty/any no. of chars;
    //         for(int ii=0; ii<=i; ii++) {
    //             if(p[ii]!='*') return 0;
    //         }
    //         return 1;
    //     }
    //     if(dp[i][j]!=-1) return dp[i][j];
    //     //Recurrence;
    //     if(p[i]==s[j] || p[i]=='?') return dp[i][j] = fn(i-1, j-1, p, s, dp);
    //     else if(p[i]=='*') {
    //         return dp[i][j] =  (fn(i-1, j, p, s, dp) // * taken as empty char and moved on;
    //         ||
    //         fn(i, j-1, p, s, dp) // * taken as the jth char
    //         );
    //     }
    //     else return dp[i][j] = 0;
    // }
    bool isMatch(string s, string p) {
        int n = p.size();
        int m = s.size();

        //Memoization:
        // vector<vector<int>> dp(n, vector<int>(m, -1));
        // return (bool)fn(n-1, m-1, p, s, dp);

        //Tabulation:
        // vector<vector<bool>> dp(n+1, vector<bool>(m+1, false));
        // //Base cases;
        // dp[0][0] = true;
        // for(int j=0; j<=m; j++) dp[0][j] = false; //Check: j starts from 1 right?
        // for(int i=0; i<=n; i++) {
        //     bool flag = true;
        //     for(int ii = 1; ii<=i; ii++) {
        //         if(p[ii-1]!='*') {
        //             flag = false;
        //             break;
        //         }
        //     }
        //     dp[i][0] = flag;
        // }
        // //Building the steps;
        // for(int i=1; i<=n; i++) {
        //     for(int j=1; j<=m; j++) {
        //         if(p[i-1]==s[j-1] || p[i-1]=='?') {
        //             dp[i][j] = dp[i-1][j-1];
        //         } 
        //         else if(p[i-1]=='*') {
        //             dp[i][j] = dp[i-1][j] // * as empty char 
        //                         || dp[i][j-1]; // * as jth char
        //         }
        //         else dp[i][j] = false;
        //     }
        // }
        // return dp[n][m];

        //Tabulation + Space Optimization:
        vector<bool> prev(m+1, false), cur(m+1, false);
        //Base cases;
        prev[0] = true;
        //Building the steps;
        for(int i=1; i<=n; i++) {
            bool flag = true;
            for(int ii=1; ii<=i; ii++) {
                if(p[ii-1]!='*') {
                    flag=false;
                    break;
                }
            }
            cur[0] = flag;
            for(int j=1; j<=m; j++) {
                if(p[i-1]==s[j-1] || p[i-1]=='?') {
                    cur[j] = prev[j-1];
                }
                else if(p[i-1]=='*') {
                    cur[j] = prev[j] // * as empty char
                            || cur[j-1]; // * as jth char
                }
                else cur[j] = false;
            }
            prev = cur;
        }
        return prev[m];
    }
};
```
**Key:** Check the recurrence relation; it will help you understand the approach.

---

Let’s conclude here, and I’ll catch you in the next one!