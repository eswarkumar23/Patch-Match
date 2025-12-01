import cv2
import numpy as np
import matplotlib.pyplot as plt
from nnf import NNF
from time import time

class InpaintNNF:
    def __init__(self, image, mask, patch_w=7, max_pm_iters=5):
        assert image.shape[1] == mask.shape[1] and image.shape[0] == mask.shape[0]
        self.image = image
        self.mask = mask
        self.patch_w = patch_w
        self.max_pm_iters = max_pm_iters
        self.pyramid = [] 
        
        self._build_pyramid()
        
    
    def _build_pyramid(self):
        self.pyramid.append((self.image, self.mask))
        img = self.image.copy()
        mask = self.mask.copy()
        stopping_patch_size = self.patch_w 
        while img.shape[0] >= stopping_patch_size and img.shape[1] >= stopping_patch_size:
            img, mask = self._downsample_img_and_mask(img, mask, 2)
            self.pyramid.append((img, mask))


    def _downsample_img_and_mask(self, img, mask, factor=2):
        blur_kernel = np.array([[1, 5, 10, 5, 1]]).astype(np.float32)
        blur_kernel = blur_kernel.T @ blur_kernel

        new_img = np.zeros((img.shape[0]//factor, img.shape[1]//factor, 3)).astype(np.float32)
        new_mask = np.ones((mask.shape[0]//factor, mask.shape[1]//factor)).astype(np.float32)

        img = np.pad(img, ((2, 2), (2, 2), (0, 0)), mode='edge').astype(np.float32)
        mask = np.pad(mask, ((2, 2), (2, 2)), mode='edge').astype(np.float32)
        
        for x in range(2, img.shape[1]-2-1, 2):
            for y in range(2, img.shape[0]-2-1, 2):
                mask_patch = mask[y-2:y+3, x-2:x+3]
                sum_non_zero = np.sum(mask_patch * blur_kernel)
                for c in range(3):
                    img_patch = img[y-2:y+3, x-2:x+3, c]
                    val = np.sum(img_patch * mask_patch * blur_kernel)
                    if sum_non_zero > 0:
                        val /= sum_non_zero
                        new_img[(y-2)//2, (x-2)//2, c] = val
                    else:
                        new_mask[(y-2)//2, (x-2)//2] = 0

        return new_img.astype(np.uint8), new_mask.astype(np.uint8)

    
    def _upsample_img(self, img, final_size):
        height, width = final_size[:2]
        return cv2.resize(img.astype(np.uint8), (width, height)).astype(np.int32)

    
    def _get_patch_similarity(self, nnf_object, ax, ay, bx, by):
        patch_dist = nnf_object.patch_distance(ax, ay, bx, by)
        return 1 - patch_dist/nnf_object.MAX_PATCH_DIFF
    
    def weighted_copy(self, src, mask, ax, ay, bx, by, w):

        votes = np.zeros((src.shape[0], src.shape[1], 4))

        if mask[ay, ax] == 0:
            return votes
        
        votes[by, bx, 0] += src[ay, ax, 0] * w
        votes[by, bx, 1] += src[ay, ax, 1] * w
        votes[by, bx, 2] += src[ay, ax, 2] * w
        votes[by, bx, 3] += w

        return votes

    def e_step(self, nnf_obj, new_src=None, new_mask=None, upscale=False):
        votes = np.zeros((new_src.shape[0], new_src.shape[1], 4))
        for ay in range(nnf_obj.ah):
            for ax in range(nnf_obj.aw):
                
                by, bx = nnf_obj.nnf[ay, ax]
                w = self._get_patch_similarity(nnf_obj, ax, ay, bx, by)

                if upscale:
                    votes += self.weighted_copy(new_src, new_mask, 2*ax, 2*ay, 2*bx, 2*by, w)
                    votes += self.weighted_copy(new_src, new_mask, 2*ax+1, 2*ay, 2*bx+1, 2*by, w)
                    votes += self.weighted_copy(new_src, new_mask, 2*ax, 2*ay+1, 2*bx, 2*by+1, w)
                    votes += self.weighted_copy(new_src, new_mask, 2*ax+1, 2*ay+1, 2*bx+1, 2*by+1, w)

                else:
                    votes += self.weighted_copy(new_src, new_mask, ax, ay, bx, by, w)

        return votes
    
    def m_step(self, new_target, votes):
        new_target = new_target.copy()
        for by in range(new_target.shape[0]):
            for bx in range(new_target.shape[1]):
                if votes[by, bx, 3] > 0:
                    new_target[by, bx, 0] = votes[by, bx, 0] / votes[by, bx, 3]
                    new_target[by, bx, 1] = votes[by, bx, 1] / votes[by, bx, 3]
                    new_target[by, bx, 2] = votes[by, bx, 2] / votes[by, bx, 3]
        return new_target


    def expectation_maximization(self, nnf_src_to_target_obj, nnf_target_to_src_obj, level, is_last_iter=False):
        nnf_src_to_target, nnf_src_to_target_dist = nnf_src_to_target_obj.compute_nnf()
        nnf_target_to_src, nnf_target_to_src_dist = nnf_target_to_src_obj.compute_nnf()
        
        upscale = False

        if is_last_iter:
            new_src = self.pyramid[level-1][0]
            new_mask = self.pyramid[level-1][1]
            new_target = self._upsample_img(nnf_src_to_target_obj.b.copy(), new_src.shape)
            upscale = True

        else:
            new_src = self.pyramid[level][0]
            new_mask = self.pyramid[level][1]
            new_target = nnf_src_to_target_obj.b.copy()


        votes1 = self.e_step(nnf_src_to_target_obj, new_src, new_mask, upscale=upscale)
        votes2 = self.e_step(nnf_target_to_src_obj, new_src, new_mask, upscale=upscale)
        votes = votes1 + votes2

        new_target = self.m_step(new_target, votes)

        return new_target, nnf_src_to_target, nnf_target_to_src
        
    def inpaint(self):

        inpainted_images = []

        num_pyramid_levels = len(self.pyramid)
        inpainted_img = None
        running_nnf_src_to_target = None
        running_nnf_target_to_src = None
        for level in range(num_pyramid_levels-1, 0, -1):
            start_time = time()

            iterEM = 1+2*level
            iterNNF = min(self.max_pm_iters, 1+level)
            src, mask = self.pyramid[level]
            no_mask = np.ones_like(mask)
            
            if level == num_pyramid_levels-1:
                inpainted_img = src.copy()
                
                nnf_src_to_target_obj = NNF(src, inpainted_img, mask_a=mask, mask_b=no_mask, patch_w=self.patch_w, pm_iters=iterNNF)
                nnf_src_to_target_obj.initialize_nnf_with_mask(mask)
            
                nnf_target_to_src_obj = NNF(inpainted_img, src, mask_a=no_mask, mask_b=mask, patch_w=self.patch_w, pm_iters=iterNNF)
                nnf_target_to_src_obj.initialize_nnf_with_mask(mask)
            else:
                nnf_src_to_target_obj = NNF(src, inpainted_img, mask_a=mask, mask_b=no_mask, patch_w=self.patch_w, pm_iters=iterNNF)
                nnf_src_to_target_obj.initialize_nnf_with_other_nnf(running_nnf_src_to_target)
                nnf_src_to_target_obj.initialize_nnf_with_mask(mask)
            
                nnf_target_to_src_obj = NNF(inpainted_img, src, mask_a=no_mask, mask_b=mask, patch_w=self.patch_w, pm_iters=iterNNF)
                nnf_target_to_src_obj.initialize_nnf_with_other_nnf(running_nnf_target_to_src)
                nnf_target_to_src_obj.initialize_nnf_with_mask(mask)
            
            for em_step in range(iterEM):
                
                inpainted_img, running_nnf_src_to_target, running_nnf_target_to_src = self.expectation_maximization(
                    nnf_src_to_target_obj, nnf_target_to_src_obj, level, is_last_iter=(em_step == iterEM-1))
                
                nnf_src_to_target_obj.b = inpainted_img.copy()
                nnf_target_to_src_obj.a = inpainted_img.copy()     
                
            inpainted_images.append(inpainted_img.astype(np.uint8))

            print("Level", level, "done in", time()-start_time, "seconds.")
            
        return inpainted_images

        