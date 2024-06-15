import os
import glob
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import argparse
from typing import List
from numpy.typing import ArrayLike


class ORBObjectDetector:
    def __init__(self, threshold: float):
        """
        Initializes the ORB object detector with a given matching threshold.

        :param threshold: The threshold for determining if an object is present based on descriptor matching.
        """
        self.orb = cv.ORB_create()  # Create an ORB detector
        self.matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)  # Create a matcher
        self.object_descriptors = []  # Initialize a list to store descriptors of the object
        self.background_descriptors = []  # Initialize a list to store descriptors of the background
        self.threshold = threshold  # Set the matching threshold

    def train(self, img: ArrayLike, present: bool) -> None:
        """
        Trains the detector with an image and its label indicating the presence of the object.

        :param img: The training image.
        :param present: Boolean indicating if the object is present in the image.
        """
        # Detect keypoints and compute descriptors using ORB
        kp, des = self.orb.detectAndCompute(np.asarray(img), None)
        if des is not None:
            if present:
                # If the object is present, store the descriptors
                self.object_descriptors.extend(des)
            else:
                # If the object is not present, store the descriptors as background
                self.background_descriptors.extend(des)

    def filter_object_descriptors(self) -> None:
        """
        Filters out the descriptors that are common with the background.
        """
        if not self.object_descriptors or not self.background_descriptors:
            return

        # Convert lists to numpy arrays
        object_descriptors = np.array(self.object_descriptors)
        background_descriptors = np.array(self.background_descriptors)

        # Use matcher to find common descriptors
        matches = self.matcher.match(object_descriptors, background_descriptors)

        # Check if matches is empty
        if not matches:
            print("No matches found with background descriptors. All object descriptors are considered unique.")
            return

        # Create a mask to filter out background descriptors from object descriptors
        mask = np.ones(len(object_descriptors), dtype=bool)
        for match in matches:
            mask[match.queryIdx] = False

        # Update object descriptors to only include unique ones
        self.object_descriptors = object_descriptors[mask].tolist()

    def classify(self, img: ArrayLike) -> (bool, List[cv.KeyPoint]):
        """
        Classifies if the object is present in the given image.

        :param img: The image to classify.
        :return: Tuple of boolean indicating if the object is present in the image and list of keypoints.
        """
        # Detect keypoints and compute descriptors for the test image
        kp, des = self.orb.detectAndCompute(np.asarray(img), None)
        if des is None:
            # If no descriptors are found, return False (object not present) and empty keypoints
            return False, []

        if not self.object_descriptors:
            # If no object descriptors are stored, return False (object not present) and empty keypoints
            return False, []

        # Convert the list of object descriptors to a numpy array
        object_descriptors = np.array(self.object_descriptors)

        # Match descriptors of the test image with stored object descriptors
        matches = self.matcher.match(des, object_descriptors)

        if not matches:
            # If no matches found, return False (object not present) and empty keypoints
            return False, []

        # Sort matches based on distance (lower distance means better match)
        matches = sorted(matches, key=lambda x: x.distance)

        # Print the number of matches and their distances
        print(f"Number of matches: {len(matches)}")
        print("Distances of matches:")
        for match in matches:
            print(match.distance)

        # Calculate the sum of matching scores
        y = sum(1 / (1 + match.distance) for match in matches)

        # Print the value of y
        print(f"Sum of matching scores (y): {y}")
        print(f"Threshold: {self.threshold}")


        # Return True if the sum of matching scores is greater than the threshold, otherwise False
        return y > self.threshold, kp


def load_images_from_folder(folder: str) -> List[ArrayLike]:
    """
    Loads all images from a specified folder.

    :param folder: The folder from which to load images.
    :return: List of loaded images.
    """
    images = []
    for extension in ['*.jpg', '*.png']:
        for filename in glob.glob(os.path.join(folder, extension)):
            img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
    return images


def main(threshold: float, object_present_images_dir: str, object_absent_images_dir: str, test_image_path: str):
    # Create an ORB object detector with a specified threshold
    detector = ORBObjectDetector(threshold=threshold)

    # Load training images from specified directories
    print("Loading training images with the object...")
    training_images_present = load_images_from_folder(object_present_images_dir)

    if not training_images_present:
        raise Exception("No training images found in the specified 'object_present_images_dir' directory.")

    print(f"Loaded {len(training_images_present)} images with the object.")

    if object_absent_images_dir:
        print("Loading training images without the object...")
        training_images_absent = load_images_from_folder(object_absent_images_dir)
        print(f"Loaded {len(training_images_absent)} images without the object.")
    else:
        training_images_absent = []
        print("No training images without the object provided.")

    # Train the detector with images containing the object
    print("Training with images containing the object...")
    for img in training_images_present:
        detector.train(img, present=True)

    # Train the detector with images not containing the object
    if training_images_absent:
        print("Training with images not containing the object...")
        for img in training_images_absent:
            detector.train(img, present=False)

        # Filter out common descriptors to get unique object descriptors
        print("Filtering object descriptors...")
        detector.filter_object_descriptors()
    else:
        print("Skipping filtering object descriptors as no background images provided.")

    # Load and classify the test image
    print(f"Classifying the test image: {test_image_path}")
    if not os.path.isfile(test_image_path):
        raise Exception(f"The test image path does not exist: {test_image_path}")

    img_test = cv.imread(test_image_path, cv.IMREAD_GRAYSCALE)
    if img_test is None:
        raise Exception(f"The specified test image could not be loaded: {test_image_path}")

    result, keypoints = detector.classify(img_test)

    # Print the classification result
    print("Classification complete.")
    print("Object present" if result else "Object not present")

    # Draw keypoints on the test image
    img_with_keypoints = cv.drawKeypoints(img_test, keypoints, None, color=(0, 255, 0), flags=0)

    # Display the test image with the keypoints
    plt.imshow(img_with_keypoints, cmap='gray')
    plt.title("Object present" if result else "Object not present")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ORB Object Detector')
    parser.add_argument('--threshold', type=float, required=True, help='Threshold for determining object presence')
    parser.add_argument('--object_present_images_dir', type=str, required=True,
                        help='Path to the folder containing images with the object')
    parser.add_argument('--object_absent_images_dir', type=str, required=False,
                        help='Path to the folder containing images without the object')
    parser.add_argument('--test_image', type=str, required=True, help='Path to the test image')

    args = parser.parse_args()

    main(args.threshold, args.object_present_images_dir, args.object_absent_images_dir, args.test_image)
