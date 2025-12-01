from time import time
import numpy as np
import cv2
import random

INF = np.inf

class NNF:
    def __init__(self, a, b, mask_a=None, mask_b=None, patch_w=7, pm_iters=5, rs_max=INF, nnf_init="Random"):
        self.patch_w = patch_w
        self.pm_iters = pm_iters
        self.rs_max = rs_max
        self.nnf_init = nnf_init

        self.MAX_RGB_DIFF = 255 * 255 * 3
        self.MAX_PATCH_DIFF = self.MAX_RGB_DIFF * patch_w * patch_w
        
        self.a = a.astype(np.int32)
        self.b = b.astype(np.int32)
        self.mask_a = mask_a
        self.mask_b = mask_b

        self.ah, self.aw = self.a.shape[0], self.a.shape[1]
        self.bh, self.bw = self.b.shape[0], self.b.shape[1]
        
        self.nnf = np.zeros((self.a.shape[0], self.a.shape[1], 2), dtype=np.int32)
        self.nnf_dist = np.zeros((self.a.shape[0], self.a.shape[1]), dtype=np.int32)
        
        if self.nnf_init == "Random":
            self.initialize_nnf()

    def _in_border(self, ax, ay):
        return ax < self.patch_w // 2 or ax >= self.aw - self.patch_w // 2 or ay < self.patch_w // 2 or ay >= self.ah - self.patch_w // 2
        
    def initialize_nnf(self):
        for ay in range(self.ah):
            for ax in range(self.aw):
                if self._in_border(ax, ay):
                    self.nnf[ay, ax] = (ay, ax)
                    self.nnf_dist[ay, ax] = 0
                else:
                    bx = random.randint(0, self.bw - 1)
                    by = random.randint(0, self.bh - 1)
                    self.nnf[ay, ax] = (by, bx)
                    self.nnf_dist[ay, ax] = self.patch_distance(ax, ay, bx, by)
                
    def initialize_nnf_with_other_nnf(self, other_nnf):
        other_nnf = other_nnf.astype(np.float32) * 2
        other_nnf_upsampled = cv2.resize(other_nnf, (self.aw, self.ah))
        other_nnf_upsampled = other_nnf_upsampled.astype(np.int32)
        self.nnf = other_nnf_upsampled.copy()
        for ay in range(self.ah):
            for ax in range(self.aw):
                if self._in_border(ax, ay):
                    self.nnf[ay, ax] = (ay, ax)
                    self.nnf_dist[ay, ax] = 0
                else:
                    by, bx = other_nnf_upsampled[ay, ax]
                    self.nnf_dist[ay, ax] = self.patch_distance(ax, ay, bx, by)
    
    def initialize_nnf_with_mask(self, mask):
        for ay in range(self.ah):
            for ax in range(self.aw):
                if mask[ay, ax] == 0 and not self._in_border(ax, ay):
                    bx = random.randint(0, self.bw - 1)
                    by = random.randint(0, self.bh - 1)
                    self.nnf[ay, ax] = (by, bx)
                    self.nnf_dist[ay, ax] = self.patch_distance(ax, ay, bx, by)
                else:
                    self.nnf[ay, ax] = (ay, ax)
                    self.nnf_dist[ay, ax] = 0
    

    def patch_distance(self, ax, ay, bx, by):
        if self._in_border(ax, ay) or self._in_border(bx, by):
            return 0
        patch_a = self.a[ay - self.patch_w // 2:ay + self.patch_w // 2 + 1, ax - self.patch_w // 2:ax + self.patch_w // 2 + 1]
        patch_b = self.b[by - self.patch_w // 2:by + self.patch_w // 2 + 1, bx - self.patch_w // 2:bx + self.patch_w // 2 + 1]
        
        ssd = np.sum((patch_a - patch_b) ** 2, axis=2)
        if self.mask_a is not None:
            mask_patch_a = self.mask_a[ay - self.patch_w // 2:ay + self.patch_w // 2 + 1, ax - self.patch_w // 2:ax + self.patch_w // 2 + 1]
            ssd = np.where(mask_patch_a == 0, self.MAX_RGB_DIFF, ssd)

        if self.mask_b is not None:
            mask_patch_b = self.mask_b[by - self.patch_w // 2:by + self.patch_w // 2 + 1, bx - self.patch_w // 2:bx + self.patch_w // 2 + 1]
            ssd = np.where(mask_patch_b == 0, self.MAX_RGB_DIFF, ssd)

        return np.sum(ssd)
    
    def improve_guess(self, ax, ay, d_best, bx_new, by_new):
        d = self.patch_distance(ax, ay, bx_new, by_new)
        if d < d_best:
            self.nnf[ay, ax] = (by_new, bx_new)
            self.nnf_dist[ay, ax] = d
    
    def propagate(self, iter_num, ax, ay, x_change, y_change):
        d_best = self.nnf_dist[ay, ax]

        if 0 <= ax - x_change < self.aw:
            y_prop, x_prop = self.nnf[ay, ax - x_change]
            x_prop += x_change
            if 0 <= x_prop < self.bw:
                self.improve_guess(ax, ay, d_best, x_prop, y_prop)

        if 0 <= ay - y_change < self.ah:
            y_prop, x_prop = self.nnf[ay - y_change, ax]
            y_prop += y_change
            if 0 <= y_prop < self.bh:
                self.improve_guess(ax, ay, d_best, x_prop, y_prop)            


    def random_search(self, ax, ay):
        rs_start = min(self.rs_max, max(self.bw, self.bh))

        mag = rs_start

        while mag >= 1:
            y_best, x_best = self.nnf[ay, ax]
            d_best = self.nnf_dist[ay, ax]
            
            x_min = max(x_best - mag, 0)
            x_max = min(x_best + mag + 1, self.bw)
            y_min = max(y_best - mag, 0)
            y_max = min(y_best + mag + 1, self.bh)

            if x_min == x_max or y_min == y_max:
                mag = mag // 2
                continue
            x_rand = random.randint(x_min, x_max - 1)
            y_rand = random.randint(y_min, y_max - 1)

            self.improve_guess(ax, ay, d_best, x_rand, y_rand)

            mag = mag // 2

    def compute_nnf(self):
        for iter_num in range(self.pm_iters):
            t_start = time()
            y_start = 0
            y_end = self.ah
            y_change = 1
            x_start = 0
            x_end = self.aw
            x_change = 1

            if iter_num % 2 == 1:
                y_start = y_end - 1
                y_end = -1
                y_change = -1
                x_start = x_end - 1
                x_end = -1
                x_change = -1

            for ay in range(y_start, y_end, y_change):
                for ax in range(x_start, x_end, x_change):
                    self.propagate(iter_num, ax, ay, x_change, y_change)
                    self.random_search(ax, ay)

            t_end = time()
        return self.nnf, self.nnf_dist


