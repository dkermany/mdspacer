import numpy as np
import cv2
import tifffile
from oiffile import OifFile

class OifImageViewer:

    # Channel 0: Tumor cells (anti-GFP antibodies)
    # Channel 1: NG2+ cells (TD-Tomato for NG2+ cree recognized with IP-antibody)
    # Channel 2: Blood vessels (CD31 & VE-Cadherin (CD144))
    
    def __init__(self, oif: OifFile) -> None:
        self._arr = oif.asarray()
        self.figsize = (25,25)
        self.md = {
            "x_step": float(oif.mainfile["Reference Image Parameter"]["WidthConvertValue"]),
            "y_step": float(oif.mainfile["Reference Image Parameter"]["HeightConvertValue"]),
            "z_step": float(oif.mainfile["Axis 3 Parameters Common"]["Interval"]),
            "x_unit": oif.mainfile["Reference Image Parameter"]["WidthUnit"],
            "y_unit": oif.mainfile["Reference Image Parameter"]["HeightUnit"],
            "z_unit": oif.mainfile["Axis 3 Parameters Common"]["PixUnit"],
        }
        # print(self)
        
    def get_image(self, ch=0, z=0, beta=255) -> np.ndarray:
        # print(self._arr.dtype)
        """Return image slice through specified z value"""
        return self.normalize(self._arr[ch][z], beta=beta)
    
    def get_array(self) -> np.ndarray:
        return self._arr
    
    def get_x_slice(self, x, ch=0, beta=255) -> np.ndarray:
        """Return image slice through specified x value"""
        return self.normalize(self._arr[ch][:, :, x], beta=beta)
    
    def get_y_slice(self, y, ch=0, beta=255) -> np.ndarray:
        """Return image slice through specified x value"""
        return self.normalize(self._arr[ch][:, y, :], beta=beta)

    def show_image(self, plt, image, color=-1, origin="upper") -> None:
        plt.figure(figsize=self.figsize)
        if color > -1:
            color_image = np.zeros((*self.img_shape, 3), dtype=np.uint16)
            color_image[:,:,color] = image
            plt.imshow(color_image)
            return
        plt.imshow(self.BGR2RGB(image))
        
    def combined_image(self, plt, z=0) -> None:
        plt.figure(figsize=self.figsize)
        color_image = np.zeros((*self.img_shape, 3), dtype=np.uint16)
        for n_ch, arr_ch in enumerate(self._arr):
            color_image[:,:,n_ch] = self.normalize(arr_ch[z])
        plt.imshow(color_image)
        
    @staticmethod
    def normalize(image, beta=255) -> np.ndarray:
        """
        Converts uint16 to uint8 by default
        0-255 for uint8
        0-65535 for uint16
        """
        dtype = "uint8" if beta == 255 else "uint16"
        return cv2.normalize(
            image, 
            dst=None, 
            alpha=0, 
            beta=beta, 
            norm_type=cv2.NORM_MINMAX
        ).astype(dtype)
    
    def BGR2RGB(self, image) -> np.ndarray:
        return cv2.cvtColor(self.normalize(image), cv2.COLOR_BGR2RGB)
    
    def __str__(self):
        cnvt_labels = [self.md["x_step"], self.md["x_unit"],
                       self.md["y_step"], self.md["y_unit"],
                       self.md["z_step"], self.md["z_unit"]]
        return (
            f"Image shape: {self._arr.shape}\n"
            "Axes: CZYX\n"
            f"Dtype: {self._arr.dtype}\n"
            "Intervals: X ({}{}) Y ({}{}) Z ({}{})\n".format(*cnvt_labels)
        )
    
    
    def save_as_tif(self, filename) -> None:
        tifffile.imwrite(filename, self._arr[2], metadata=self.md)
        
    def save_as_tif_sequence(self, image, filename) -> None:
        for c in range(image.shape[0]):
            for z in range(image.shape[1]):
                tifffile.imwrite(f"{filename}_C{c}_Z{z}.tif", image[c][z], metadata=self.md)
        
