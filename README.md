# yolo-onnx-cv
### arguments
```sh
-url <file or url: string>
-cam <device: int>
-w <width: int>
-h <height: int>
-cfg <path to darknet cfg file>
-weights <path to darknet weights file>
-onnx <path to onnx yolo file>
-scaled # if set ouput model rectangles scaled to (width x height) else to (1 x 1)
```

### windows
- cmake
- opencv libraries (build with cuda for acceleration)
- put ffmpeg libraries to the folder "3rd/".

### linux
- not tested

for using onnx model converted from pytorch (yolov5, yolov7 for example) need to simplify onnx (opset 11 better):
- [Torch interpolate's onnx graph fails while parsing #23730](https://github.com/opencv/opencv/issues/23730)

Example code:
```python
import onnx
from onnx import shape_inference
from onnxsim import simplify

print("\t\t ---> READING ONNX FILE IN ONNX <---")
onnx_model = onnx.load(full_model_path)
onnx.checker.check_model(onnx_model)

onnx_model = shape_inference.infer_shapes(onnx_model)
onnx.save(onnx_model, full_model_path)

############################################
## =======> Simplify ONNX graph <======== ##
onnx_model, check = simplify(onnx_model)
onnx.save(onnx_model, full_model_path)
```
