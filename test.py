# import torch
# from torch import nn
import numpy as np


# m = nn.Conv1d(16, 36, 3, stride=2, groups=4)
# input = torch.randn(20, 16, 50)
# output = m(input)
# print(output.shape)

class Solution:
    def solve(self, nums=[2, 3, 5]):
        self.min = 13560
        self.target = 13560
        self.path = []
        self.dfs(nums)

    def dfs(self, nums=[2, 3, 5], m=1, path = None):
        if (m > 10*self.target):
            return
        if (np.abs(self.target - m) < self.min):
            print(m, ": ", self.path)
            self.min = np.abs(self.target - m)
        for x in nums:
            self.dfs(nums, m*x, self.path.append(x))
            self.path.pop()

if __name__ == '__main__':
    s = Solution()
    s.solve()
