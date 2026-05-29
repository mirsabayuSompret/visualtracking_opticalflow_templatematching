
import cv2
import numpy as np

class optical_flow:
    def __init__(self):
        print("Extracting optical flow features...")
        self.LK_Levels = 3
        self.LK_Win_Half = 7
        self.LK_Sigma = 1.0
        self.SOBEL_DIFF = np.array([1, 0, -1], dtype=np.float32)
        self.SOBEL_SMOOTH = np.array([1, 2, 1], dtype=np.float32)
    
    def extract_features(self, images) -> np.ndarray:
        motion_maps = []
        for i in range(len(images) - 1):
            current_frame = images[i][0]
            next_frame = images[i][1]
            motion_map = self._compute_motion_map(current_frame, next_frame)
            motion_maps.append(motion_map)

        return motion_maps

    def _compute_motion_map(self, current_frame, next_frame):
        current_frame = self._gaussian_blur(current_frame).astype(np.float32)
        next_frame = self._gaussian_blur(next_frame).astype(np.float32)
        pyr_current = [current_frame]
        pyr_next = [next_frame]
        for _ in range(self.LK_Levels - 1):
            pyr_current.append(self._downsample(pyr_current[-1]))
            pyr_next.append(self._downsample(pyr_next[-1]))
        h0, w0 = pyr_current[-1].shape
        u = np.zeros((h0, w0), dtype=np.float32)
        v = np.zeros((h0, w0), dtype=np.float32)
        for level in range(self.LK_Levels - 1, -1, -1):
            lp = pyr_next[level]
            lc = pyr_current[level]
            lh, lw = lp.shape
            if level < self.LK_Levels - 1:
                u = self._upsample(u, lh, lw)
                v = self._upsample(v, lh, lw)
            warped_c = self._warp_frame(lc, u, v)
            du, dv = self._lk_flow_single_scale(lp, warped_c, win_half=self.LK_Win_Half)
            u += du 
            v += dv
        magnitude = np.sqrt(u**2 + v**2)
        mag_max = magnitude.max()
        if mag_max > 0: magnitude /= mag_max
        return magnitude.astype(np.float32)

    def _gaussian_blur(self, image):
        radius = max(1, int(3 * self.LK_Sigma))
        kernel = self._gaussian_kernel_1d(self.LK_Sigma, radius)
        return self._separable_convolve(image, kernel, kernel)
    
    def _gaussian_kernel_1d(self, sigma, radius):
        x = np.arange(-radius, radius + 1, dtype=np.float32)
        g = np.exp(-0.5 * (x / sigma) ** 2)
        return g / g.sum()

    def _separable_convolve(self, image, kernel_x, kernel_y):
        return self._convolve2d(self._convolve2d(image, kernel_x.reshape(1, -1)), kernel_y.reshape(-1, 1))
    
    def _convolve2d(self, image, kernel):
        kh, kw = kernel.shape
        ph, pw = kh // 2, kw // 2
        padded  = np.pad(image, ((ph, ph), (pw, pw)), mode="reflect")
        shape   = image.shape + kernel.shape
        strides = padded.strides + padded.strides
        patches = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
        return (patches * kernel).sum(axis=(-2, -1))
    
    def _downsample(self, image):
        h2, w2 = image.shape[0] // 2, image.shape[1] // 2
        return 0.25 * (image[:h2*2:2, :w2*2:2] + image[1:h2*2:2, :w2*2:2]
                 + image[:h2*2:2, 1:w2*2:2] + image[1:h2*2:2, 1:w2*2:2])
    
    def _upsample(self, image, height, width):
        h, w = image.shape
        yi = np.clip((np.arange(height) / (height / h)).astype(int), 0, h - 1)
        xi = np.clip((np.arange(width) / (width / w)).astype(int), 0, w - 1)
        return image[np.ix_(yi, xi)] * 2.0
    
    def _warp_frame(self, frame, u, v):
        h, w = frame.shape
        yy, xx = np.meshgrid(np.arange(h, dtype=np.float32),
                            np.arange(w, dtype=np.float32), indexing="ij")
        src_y = np.clip(yy + v, 0, h - 1); src_x = np.clip(xx + u, 0, w - 1)
        y0 = src_y.astype(int); x0 = src_x.astype(int)
        y1 = np.clip(y0 + 1, 0, h - 1);   x1 = np.clip(x0 + 1, 0, w - 1)
        fy = src_y - y0; fx = src_x - x0
        return (frame[y0,x0]*(1-fy)*(1-fx) + frame[y1,x0]*fy*(1-fx)
            + frame[y0,x1]*(1-fy)*fx     + frame[y1,x1]*fy*fx).astype(np.float32)
    
    def _lk_flow_single_scale(self, next_frame, current_frame, win_half):
        h, w  = current_frame.shape
        avg   = (current_frame + next_frame) * 0.5
        Ix    = self._spatial_gradient_x(avg)
        Iy    = self._spatial_gradient_y(avg)
        It    = next_frame.astype(np.float32) - current_frame.astype(np.float32)
        win   = 2 * win_half + 1
        box_k = np.ones(win, dtype=np.float32)

        def _box(a): return self._separable_convolve(a, box_k, box_k)

        sIxx = _box(Ix * Ix); sIyy = _box(Iy * Iy); sIxy = _box(Ix * Iy)
        sIxt = _box(Ix * It); sIyt = _box(Iy * It)
        det  = sIxx * sIyy - sIxy * sIxy
        mask = np.abs(det) > 1e-6
        u = np.zeros((h, w), dtype=np.float32)
        v = np.zeros((h, w), dtype=np.float32)
        u[mask] = (-sIxt[mask] * sIyy[mask] + sIyt[mask] * sIxy[mask]) / det[mask]
        v[mask] = (-sIyt[mask] * sIxx[mask] + sIxt[mask] * sIxy[mask]) / det[mask]
        return u, v

    def _spatial_gradient_x(self, img): return self._separable_convolve(img, self.SOBEL_DIFF, self.SOBEL_SMOOTH)
    def _spatial_gradient_y(self, img): return self._separable_convolve(img, self.SOBEL_SMOOTH, self.SOBEL_DIFF)

class template_matching:
    def __init__(self, templateImage):
        self.templateImage = templateImage
        self.GRID_ROWS = 8
        self.GRID_COLS = 8
        self.SEARCH_MARGIN = 20
        print("Extracting template matching features...")
    
    def extract(self, images) -> np.ndarray:
        motion_maps = []
        for i in range(len(images) - 1):
            current_frame = images[i]
            next_frame = images[i + 1]
            motion_map = self._compute_motion_map(current_frame, next_frame, self.SEARCH_MARGIN)
            motion_maps.append(motion_map)
        return motion_maps
    
    def _compute_motion_map(self, current_frame : np.ndarray, next_frame: np.ndarray, margin: int) -> np.ndarray :
        """Grid-of-patches NCC template matching → normalised [0,1] motion magnitude."""
        h, w = current_frame.shape[:2]
        motion_map = np.zeros((h, w), dtype=np.float32)
        ph = h // self.GRID_ROWS; pw = w // self.GRID_COLS
        if ph < 4 or pw < 4: return motion_map
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                y1, y2 = r*ph, (r+1)*ph; x1, x2 = c*pw, (c+1)*pw
                template = current_frame[y1:y2, x1:x2]
                sy1 = max(0, y1-margin); sy2 = min(h, y2+margin)
                sx1 = max(0, x1-margin); sx2 = min(w, x2+margin)
                search = next_frame[sy1:sy2, sx1:sx2]
                if search.shape[0] < template.shape[0] or search.shape[1] < template.shape[1]:
                    continue
                result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
                _, _, _, max_loc = cv2.minMaxLoc(result)
                dy = (sy1 + max_loc[1]) - y1; dx = (sx1 + max_loc[0]) - x1
                motion_map[y1:y2, x1:x2] = np.sqrt(dx**2 + dy**2)
        max_disp = motion_map.max()
        if max_disp > 0: motion_map /= max_disp
        return motion_map
