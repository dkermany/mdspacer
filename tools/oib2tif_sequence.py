import argparse
import os
import tifffile
from oiffile import OifFile

class OifImageViewer:
    # Channel 0: Tumor cells (anti-GFP antibodies)
    # Channel 1: NG2+ cells (TD-Tomato for NG2+ cree recognized with IP-antibody)
    # Channel 2: Blood vessels (CD31 & VE-Cadherin (CD144))

    def __init__(self, oif: OifFile) -> None:
        self._arr = oif.asarray()
        self.md = {
            "x_step": oif.mainfile["Reference Image Parameter"]["WidthConvertValue"],
            "y_step": oif.mainfile["Reference Image Parameter"]["HeightConvertValue"],
            "z_step": oif.mainfile["Axis 3 Parameters Common"]["Interval"],
            "x_unit": oif.mainfile["Reference Image Parameter"]["WidthUnit"],
            "y_unit": oif.mainfile["Reference Image Parameter"]["HeightUnit"],
            "z_unit": oif.mainfile["Axis 3 Parameters Common"]["PixUnit"],
        }

    def save_as_tif(self, filename: str) -> None:
        tifffile.imwrite(filename, self._arr, metadata=self.md)

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

def validate_path(path: str) -> str:
    if os.path.exists(path) and path.endswith((".oib", ".oif")):
        return os.path.normpath(path)
    raise ValueError(f"Invalid path or type. OIB and OIF only. Got: {path}")

def main():
    input_path = validate_path(FLAGS.input_path)
    output_path = os.path.splitext(input_path)[0] + ".tif"
    with OifFile(input_path) as oif:
        viewer = OifImageViewer(oif)
    print(f"Input: {input_path}\n{viewer}")
    print(f"Output: {output_path}")
    print("Conversion begins...")
    viewer.save_as_tif(output_path)
    print("OIB/OIF -> TIF Conversion successfully completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        required=True,
        type=str,
        help="Path to .oib/.oif stitched file"
    )
    FLAGS, _ = parser.parse_known_args()
    main()

