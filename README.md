# PJ_WMA_ORB_Object_Detector

This application uses ORB (Oriented FAST and Rotated BRIEF) for object detection. It allows you to train a detector with images containing an object and classify if the object is present in a test image based on a specified threshold.

## Requirements

See conda.yml for required dependencies. Useful commands:

```
conda env create -f conda.yml
```

```
conda env update -f conda.yml
```

```
conda activate pja-wma
```

## Application

Run the script with the following command:

```
python object_detector.py \
--threshold 0.7 \
--object_present_images_dir data/object_images/present \
--test_image data/object_images/test_image.jpg
```

- `--threshold`: the threshold for determining object presence.
- `--object_present_images_dir`: path to the folder containing images with the object.
- `--object_absent_images_dir`: path to the folder containing images without the object (optional)
- `--test_image`: path to the test image.
