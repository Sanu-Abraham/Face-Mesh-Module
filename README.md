# Face Mesh Module using Mediapipe

This module provides a Python implementation of face mesh generation using the Mediapipe library and OpenCV. It enables the generation of a detailed face mesh on detected faces in images or video streams, allowing for applications such as facial analysis, 3D modeling, and more.

## Dependencies

The following dependencies are required to use the module:

- OpenCV
- Mediapipe

You can install these dependencies using pip:

```shell
pip install opencv-python mediapipe
```

## Usage

To use the Face Mesh module, import it in your Python code:

```python
from face_mesh import FaceMesh
```

Then, create an instance of the `FaceMesh` class with your desired configuration:

```python
mesh = FaceMesh(mode=False, maxFaces=3, refineLandmarks=False, detectConf=0.5, trackConf=0.5)
```

You can adjust the parameters according to your specific requirements. Once the `FaceMesh` object is created, you can call its methods to generate face meshes on detected faces in images or video frames.

For example, to generate face meshes on a single image:

```python
import cv2

image = cv2.imread("image.jpg")
meshed_image = mesh.createFaceMesh(image)
face_landmarks, meshed_image = mesh.getPosition(image)
# Process the meshed image and face landmarks as needed
```

Refer to the script for more details and examples on how to use the `FaceMesh` class.

## License

This module is licensed under the [MIT License](LICENSE).

## Acknowledgements

The face mesh generation functionality in this module is based on the Mediapipe library, which provides the face mesh model used for generating detailed face meshes.
