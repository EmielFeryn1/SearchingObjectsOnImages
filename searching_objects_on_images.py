import cv2
import lancedb
import numpy as np
import open_clip
import os
from PIL import Image
import pandas as pd
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import uuid
import shutil
import logging
import time


class ObjectImageSearcher:
    def __init__(self, sam_model_path, clip_model_name):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        start_time = time.time()
        self.SAM = sam_model_registry["vit_h"](checkpoint=sam_model_path)
        initialization_duration = time.time() - start_time
        self.logger.info(f"SAM model initialized in {initialization_duration:.2f} seconds")

        start_time = time.time()
        self.database = lancedb.connect("data/sample-lancedb")
        database_initialization_duration = time.time() - start_time
        self.logger.info(f"Database initialized in {database_initialization_duration:.2f} seconds")

        start_time = time.time()
        self.clip = None
        self.pre_process_clip = None
        self.tokenizer = None
        self.load_clip_model(clip_model_name)
        model_loading_duration = time.time() - start_time
        self.logger.info(f"CLIP model loaded in {model_loading_duration:.2f} seconds")

        self.current_table_id = None
        self.current_image_id = None

    def load_clip_model(self, clip_model_name):
        # Create the CLIP model using OpenCLIP library
        # The pretrained argument specifies the specific pre-trained model to load
        model, _, preprocess = open_clip.create_model_and_transforms(clip_model_name, pretrained='laion2b_s34b_b79k')

        # Set the CLIP model, preprocessing function, and tokenizer
        self.clip = model
        self.pre_process_clip = preprocess
        self.tokenizer = open_clip.get_tokenizer(clip_model_name)

    def index_images_to_lancedb(self, img_path):
        """
            Indexes the segmentation to a LanceDB table by performing the following steps:

            1. Reads the source image using OpenCV.
            2. Retrieves segmentations from the image using the 'get_segmentations_from_image' method.
            3. Generates a unique identifier for the current image.
            4. Creates a directory with the generated identifier to store temporary files.
            5. Extracts and saves segmentations from the image, using 'extracting_and_saving_segmentations'.
            6. Creates or updates the LanceDB table with the extracted segmentations.
            7. Removes the temporary directory created earlier.

            Parameters:
            - img_path (str): The file path to the image to be indexed in LanceDB.
        """
        source_img = cv2.imread(img_path)

        segmentations = self.get_segmentations_from_image(img_path)

        self.current_image_id = uuid.uuid4().hex[:6]
        os.makedirs(self.current_image_id, exist_ok=True)

        self.extracting_and_saving_segmentations(segmentations, source_img)
        self.creating_lancedb_table(segmentations)

        shutil.rmtree(self.current_image_id)

    def extracting_and_saving_segmentations(self, segmentations, source_img):
        """
            Extracts and saves individual segmentations from the source image.

            Parameters:
            - segmentations (list): A list of dictionaries containing segmentation information, each with 'bbox' and 'segmentation'.
            - source_img (numpy.ndarray): The source image from which segmentations are extracted.

            Iterates through each segmentation in the provided list, extracts the corresponding region from the source image,
            saves the cropped image, computes embeddings, and updates the segmentation row.
        """
        for index, seg in enumerate(segmentations):
            cropped_img = self.extract_segmentation_from_host_image(image=source_img, bbox=seg['bbox'],
                                                                    segmentation=seg['segmentation'])

            cropped_img_path = self.current_image_id + '/{}.jpg'.format(index)
            cv2.imwrite(cropped_img_path, cropped_img)
            embeddings = self.get_image_embeddings_from_path(cropped_img_path)
            self.updating_segmentation_row(cropped_img_path, embeddings, seg)

    def creating_lancedb_table(self, segmentations):
        """
            Creates a LanceDB table with segmentation information.

            Parameters:
            - segmentations (list): A list of dictionaries containing segmentation information.
        """
        seg_df = pd.DataFrame(segmentations)
        seg_df = seg_df[
            ['img_path', 'embeddings', 'bbox', 'stability_score', 'predicted_iou', 'segmentation', 'seg_shape']]
        seg_df = seg_df.rename(columns={"embeddings": "vector"})
        self.current_table_id = self.database.create_table("table_{}".format(self.current_image_id), data=seg_df)

    @staticmethod
    def updating_segmentation_row(cropped_img_path, embeddings, seg):
        """
            Updates a segmentation row with the provided image path, embeddings, and segmentation information.

            Parameters:
            - cropped_img_path (str): The file path of the cropped image.
            - embeddings (numpy.ndarray): The embeddings computed for the cropped image.
            - seg (dict): A dictionary containing segmentation information.

            Updates the 'embeddings', 'img_path', 'seg_shape', and 'segmentation' fields in the given segmentation dictionary.
        """
        seg['embeddings'] = embeddings
        seg['img_path'] = cropped_img_path
        seg['seg_shape'] = seg['segmentation'].shape
        seg['segmentation'] = seg['segmentation'].reshape(-1)

    def get_image_embeddings_from_path(self, file_path):
        """
           Retrieves embeddings for an image from the specified file path.

           Parameters:
           - file_path (str): The file path of the image.

           Returns:
           - numpy.ndarray: The computed embeddings for the image.
           """
        input_image = Image.open(file_path)
        image = self.pre_process_clip(input_image).unsqueeze(0)

        embeddings = self.get_embeddings_by_encoding_image(image).squeeze()
        return embeddings.detach().numpy()

    def get_embeddings_by_encoding_image(self, image):
        """
            Computes embeddings for the given image using the CLIP model.

            Parameters:
            - image (torch.Tensor): The input image tensor.

            Returns:
            - torch.Tensor: The computed embeddings for the image.
        """
        with torch.no_grad():
            embeddings = self.clip.encode_image(image)
        return embeddings

    def get_segmentations_from_image(self, img_path):
        """
            Retrieves segmentations (masks) from the given image using the SAM Automatic Mask Generator.

            Parameters:
            - img_path (str): The file path of the input image.

            Returns:
            - list: A list of masks representing segmentations extracted from the image.
        """
        start_time = time.time()
        input_img = self.load_image(img_path)
        masks = self.get_masks(input_img)
        self.logger.info(f"Segmentations extracted in {time.time()-start_time:.2f} seconds")
        return masks

    def get_masks(self, input_img):
        """
           Generates masks (segmentations) using the SAM Automatic Mask Generator.
           These masks represent all detected objects inside an image.

           Parameters:
           - input_img (numpy.ndarray): The input image for segmentation.

           Returns:
           - list: A list of masks representing segmentations generated from the input image.
        """
        mask_generator = SamAutomaticMaskGenerator(self.SAM)
        masks = mask_generator.generate(input_img)
        return masks

    @staticmethod
    def load_image(img_path):
        """
            Loads an image from the specified file path.

            Parameters:
            - img_path (str): The file path of the image.

            Returns:
            - numpy.ndarray: The loaded image in RGB format.
        """
        input_img = cv2.imread(img_path)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        return input_img

    def extract_segmentation_from_host_image(self, image, bbox, segmentation):
        """
          Extracts the segmented region from the host image based on bounding box and segmentation.
          We need to separately embed all segmentations to store in vector database.

          Parameters:
          - image (numpy.ndarray): The host image from which segmentation is extracted.
          - bbox (tuple): Bounding box coordinates (x, y, width, height).
          - segmentation (numpy.ndarray): Segmentation mask corresponding to the region of interest.

          Returns:
          - numpy.ndarray: The extracted segmentation from the host image.
        """
        masked_white_canvas = self.convert_all_pixels_to_white_except_segmentation(image, segmentation)
        x, y, w, h = bbox
        segmentation = masked_white_canvas[y:y + h, x:x + w]
        return segmentation

    @staticmethod
    def convert_all_pixels_to_white_except_segmentation(image, segmentation):
        """
            Converts all pixels in the image to white except for the region specified by the segmentation.
            This function is needed to cut out the segmentation from the host image.

            Parameters:
            - image (numpy.ndarray): The original image.
            - segmentation (numpy.ndarray): The segmentation mask specifying the region of interest.

            Returns:
            - numpy.ndarray: An image with all pixels set to white except for the segmented region.
        """
        binary_mask = segmentation.astype(int)
        white_canvas = np.ones_like(image) * 255
        masked_white_canvas = np.where(binary_mask[..., None] > 0, image, white_canvas)
        return masked_white_canvas

    @staticmethod
    def flatten_list(lst):
        """
            Flattens a nested list into a single-dimensional list.

            Parameters:
            - lst (list): The nested list to be flattened.

            Returns:
            - list: The flattened list.
        """
        return [item for sublist in lst for item in sublist]

    def search_image(self, img_path, user_query,):
        """
            Searches for the closest segmentation match in the current LanceDB table based on user input.

            Parameters:
            - img_path (str): The file path of the image for visualization.
            - user_query (str): The user's text query for searching relevant segmentations.

            Retrieves the text embedding from the user query, searches the LanceDB table for the closest match,
            and displays the closest segmentation on the provided image.
        """
        embedding_list = self.translate_text_to_embedding(user_query)

        closest_match = self.matching_text_embedding_with_closest_segmentation(embedding_list)
        self.show_closest_segmentation_on_image(img_path, closest_match)

    @staticmethod
    def show_closest_segmentation_on_image(img_path, target):
        """
            Displays the closest segmentation on the input image.

            Parameters:
            - img_path (str): The file path of the image.
            - target (pandas.DataFrame): The result containing the closest segmentation information.

            Writes the processed image with highlighted segmentation to 'processed.jpg' and displays it.
        """
        segmentation_mask = cv2.convertScaleAbs(
            target.iloc[0]['segmentation'].reshape(target.iloc[0]['seg_shape']).astype(int))
        dilated_mask = cv2.dilate(segmentation_mask, np.ones((10, 10), np.uint8), iterations=1)

        surroundings_mask = dilated_mask - segmentation_mask
        highlighted_image = cv2.imread(img_path)
        highlighted_image[surroundings_mask > 0] = [253, 218, 13]

        cv2.imwrite('processed.jpg', highlighted_image)
        img = Image.open('processed.jpg')
        img.show()

    def matching_text_embedding_with_closest_segmentation(self, embedding_list):
        """
            Searches the LanceDB table for the closest segmentation match based on the given text embedding.

            Parameters:
            - embedding_list (list): The list of text embeddings from user input.

            Returns:
            - pandas.DataFrame: The result containing information about the closest segmentation.
        """
        target = self.current_table_id.search(embedding_list).limit(1).to_df()
        return target

    def translate_text_to_embedding(self, user_query):
        """
            Translates user text query into text embeddings using CLIP model.

            Parameters:
            - user_query (str): The user's text query.

            Returns:
            - list: The list of text embeddings.
        """
        tokenized_text = self.tokenizer(user_query)
        embedding = self.clip.encode_text(tokenized_text).tolist()
        embedding_list = self.flatten_list(embedding)
        return embedding_list


if __name__ == '__main__':
    sam_model_path = "sam_vit_h_4b8939.pth"
    clip_model_name = 'ViT-B-32'
    o = ObjectImageSearcher(sam_model_path, clip_model_name)
    o.index_images_to_lancedb('lake.jpg')
    o.search_image('lake.jpg', 'a lake')