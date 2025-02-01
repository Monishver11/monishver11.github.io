---
layout: post
title: DP on Squares (Patterns and Problems)
date: 2025-01-25 20:07:00-0400
featured: false
description: A structured reference set of DP on Squares problems, directly adapted from Striver's DP playlist.
tags: DP Code
categories: DSA
giscus_comments: false
related_posts: false
# toc:
#   sidebar: left
---

#### **Problems:**

[Maximal Rectangle](https://leetcode.com/problems/maximal-rectangle/)
```c++
class Solution {
public:
    int largeHist(vector<int> &heights) {
        //To understand this(if you forgot how it works), refer to that
        //problem. Comments are added to it.
        int n = heights.size();
        stack<int> st;
        vector<int> left(n, 0), right(n, 0);

        //1st pass for left;
        for(int i=0; i<n; i++) {
            while(!st.empty() && heights[st.top()] >= heights[i]) {
                st.pop();
            }
            if(st.empty()) left[i] = 0;
            else left[i] = st.top()+1;

            st.push(i);
        }

        //Clear the stack before using it for next pass;
        while(!st.empty()) st.pop();

        //2nd pass for right;
        for(int i=n-1; i>=0; i--) {
            while(!st.empty() && heights[st.top()] >= heights[i]) {
                st.pop();
            }
            if(st.empty()) right[i] = n-1;
            else right[i] = st.top()-1;

            st.push(i);
        }

        //Print - Debug;
        for(int i=0; i<n; i++) {
            cout<<left[i]<<":"<<right[i]<<endl;
            cout<<heights[i]<<endl;
        }

        //Calucate the maxArea using the right and left vector values;
        int maxArea = INT_MIN;
        for(int i=0; i<n; i++) {
            maxArea = max(maxArea, (right[i]-left[i]+1)*heights[i]);
        }
        cout<<endl;
        return maxArea;
    }
    int maximalRectangle(vector<vector<char>>& matrix) {
        //Using a concept and approach from the problem of 
        //Largest rectangle in a Histogram to build this solution;

        int n = matrix.size();
        int m = matrix[0].size();
        vector<int> heights(m, 0);
        int maxRect = INT_MIN;
        for(int i=0; i<n; i++) {
            for(int j=0; j<m; j++) {
                if(i==0) {
                    if(matrix[i][j]=='1') {
                        heights[j] = 1;
                    }
                    else {
                        heights[j] = 0;
                    }
                }
                else {
                    if(matrix[i][j]=='1') {
                        heights[j] +=1;
                    }
                    else {
                        heights[j]=0;
                    }
                }
            }
            int maxHistArea = largeHist(heights);
            maxRect = max(maxRect, maxHistArea);
        }
        return maxRect;
    }
};
```
**Key:** Refer to the [Link1](https://leetcode.com/problems/largest-rectangle-in-histogram/description/) and [Link2](https://www.youtube.com/watch?v=X0X6G-eWgQ8&t=1263s)


[Count Square Submatrices with All Ones](https://leetcode.com/problems/count-square-submatrices-with-all-ones/)
```c++
class Solution {
public:
    int countSquares(vector<vector<int>>& matrix) {
        //Directyl via Tabulation method:
        int n = matrix.size();
        int m = matrix[0].size();
        vector<vector<int>> dp(n, vector<int>(m, 0));
        int sum=0;

        for(int j=0; j<m; j++) {
            dp[0][j] = matrix[0][j];
            sum+=dp[0][j];
        }
        for(int i=1; i<n; i++) {
            dp[i][0] = matrix[i][0];
            sum+=dp[i][0];
        }
        cout<<sum;
        for(int i=1; i<n; i++) {
            for(int j=1; j<m; j++) {
                if(matrix[i][j]==1) {
                    dp[i][j] = min(dp[i-1][j], 
                        min(dp[i][j-1], dp[i-1][j-1])) + 1;
                }
                else {
                    dp[i][j] = 0;
                }
                sum+=dp[i][j];
            }
        }
        return sum;
    }
};
```
**Key:** To count square submatrices with all ones, think of each cell as the bottom-right corner of a square. The size of the square ending at that cell is determined by the smallest square that can be extended from its top, left, and top-left neighbors, plus one.

---

Let’s conclude here, and I’ll catch you in the next one!