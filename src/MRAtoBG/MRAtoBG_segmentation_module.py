import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress all warnings and babble

import nibabel as nib
import numpy as np

from tensorflow.keras.models import load_model
import tensorflow as tf
import json
import time

class Segmentation():
    def __init__(self, subj_ix, tag, images_path, segmentations_path, image_to_segment_suffix, segmentation_model_path, segmentation_model_threshold, verbose):
        """
        """
        self.subj_ix = subj_ix
        self.tag = tag
        self.images_path = images_path
        self.segmentations_path = segmentations_path
        self.image_to_segment_suffix = image_to_segment_suffix
        self.segmentation_model_path = segmentation_model_path
        self.segmentation_model_threshold = segmentation_model_threshold
        self.verbose = verbose

        segmentations_path_subj = f"{segmentations_path}/{tag}"
        tf_path = f"{segmentations_path}/{tag}/tf"
        if not os.path.exists(segmentations_path_subj):
            os.mkdir(segmentations_path_subj)
        if not os.path.exists(tf_path):
            os.mkdir(tf_path)

        subj_stem = f"{tag}_{image_to_segment_suffix}"
        self.tfrecords_path = f"{segmentations_path}/{tag}/tf/{subj_stem}.tfrecords"
        self.tfrecords_json_path = f"{segmentations_path}/{tag}/tf/{subj_stem}.json"

        self.image_path = f'{images_path}/{tag}/anat/{tag}_{image_to_segment_suffix}.nii.gz'

        self.save_path_recon_image = f'{segmentations_path}/{tag}/{tag}_{image_to_segment_suffix}_img.nii.gz'
        self.save_path_recon_label = f'{segmentations_path}/{tag}/{tag}_{image_to_segment_suffix}_lab.nii.gz'
        self.save_path_recon_pred = f'{segmentations_path}/{tag}/{tag}_{image_to_segment_suffix}_seg.nii.gz'
        self.save_path_recon_logits = f'{segmentations_path}/{tag}/{tag}_{image_to_segment_suffix}_logits.nii.gz'


    def decode_image(self, example, token, patch_dim, dtype, improper_standardisation, do_intermediate_priors):
        image = tf.io.decode_raw(example[token], dtype)

        if patch_dim[2] == 1:  # 2D
            image.set_shape([patch_dim[0] * patch_dim[1]])
            image = tf.reshape(image, (patch_dim[0], patch_dim[1], 1))
        elif patch_dim[2] > 1:  # 3D
            if improper_standardisation:
                image.set_shape([patch_dim[0] * patch_dim[1] * patch_dim[2]])
                image = tf.reshape(image, (patch_dim[0], patch_dim[1], patch_dim[2],
                                           1))  # add extra 4th unity dimension here to improperly standardise later
            else:
                image.set_shape([patch_dim[0] * patch_dim[1] * patch_dim[2]])
                image = tf.reshape(image, (patch_dim[0], patch_dim[1], patch_dim[2]))
        else:
            raise ValueError("Patch_dim must be either of length 2 or 3 (2D or 3D patch)")

        return image

    def parse_data(self, serialised_example, image_patch_dim, label_patch_dim, do_standardise=True,
                   improper_standardisation=False, do_intermediate_priors=False):
        feature = {'label': tf.io.FixedLenFeature([], tf.string), 'image': tf.io.FixedLenFeature([], tf.string)}
        example = tf.io.parse_single_example(serialised_example, feature)

        label = self.decode_image(example, 'label', label_patch_dim, tf.uint8, improper_standardisation,
                                  do_intermediate_priors)
        image = self.decode_image(example, 'image', image_patch_dim, tf.float64, improper_standardisation,
                                  do_intermediate_priors)

        if do_standardise:
            image = tf.image.per_image_standardization(image)
        image = tf.image.convert_image_dtype(image, tf.float32)

        # improper_standardisation path already added extra 4th unity dimension in decode_image, so only process proper standardisation case
        if image_patch_dim[2] > 1 and label_patch_dim[
            2] > 1 and not improper_standardisation:  # 3D image, 3D label
            image = tf.reshape(image, (image_patch_dim[0], image_patch_dim[1], image_patch_dim[2], 1))
            label = tf.reshape(label, (label_patch_dim[0], label_patch_dim[1], label_patch_dim[2], 1))
        elif image_patch_dim[2] > 1 and label_patch_dim[
            2] == 1 and not improper_standardisation:  # 3D image, 2D label
            image = tf.reshape(image, (image_patch_dim[0], image_patch_dim[1], image_patch_dim[2], 1))
        # 2D label already has 4th unity dimension added in decode_image

        return image, label

    def reconstruct_patch_dataset(self, dataset, image_patch_dim, image_size, image_patch_increment, padding, predictions=False):
        sub_st = 16
        sub_en = 48
        image_patches = []
        label_patches = []
        pred_patches = []
        logit_patches = []
        # print(predictions.shape)
        if type(predictions) != bool:  # model prediction
            for prediction in predictions:
                # print(prediction.shape)
                # prediction_argmaxed = np.argmax(prediction, axis=-1)
                # prediction_argmaxed = np.expand_dims(prediction_argmaxed, axis=-1)
                # pred_patches.append(prediction_argmaxed)
                prediction_softmaxed = tf.nn.softmax(prediction, axis=-1).numpy()
                prediction_softmaxed_thresholded = prediction_softmaxed[:, :, :, 1] > self.segmentation_model_threshold
                prediction_softmaxed_thresholded = np.expand_dims(prediction_softmaxed_thresholded, axis=-1)
                pred_patches.append(prediction_softmaxed_thresholded)
                logit_patches.append(np.expand_dims(prediction[:, :, :, 1], axis=-1))
        for image, label in dataset.unbatch().as_numpy_iterator():
            image_patches.append(image)
            label_patches.append(label)
        # print(len(image_patches), image_patches[0].shape, len(label_patches), image_size, image_patch_increment)
        c = 0

        n_x_slices = (image_size[0] - image_patch_dim[0]) // image_patch_increment[0] + 1
        n_y_slices = (image_size[1] - image_patch_dim[1]) // image_patch_increment[1] + 1
        n_z_slices = (image_size[2] - image_patch_dim[2]) // image_patch_increment[2] + 1
        n_samples = n_x_slices * n_y_slices * n_z_slices

        final_x_slice_start_idx = (n_x_slices - 1) * image_patch_increment[0]
        final_y_slice_start_idx = (n_y_slices - 1) * image_patch_increment[1]
        final_z_slice_start_idx = (n_z_slices - 1) * image_patch_increment[2]
        # print(n_x_slices, n_y_slices, n_z_slices, n_samples)
        # print(len(image_patches), len(label_patches), len(pred_patches), len(logit_patches),)

        recon_image = np.zeros(image_size)
        recon_label = np.zeros(image_size)
        recon_pred = np.zeros(image_size)
        recon_logits = np.zeros(image_size)
        for z_slice_start in range(0, final_z_slice_start_idx + 1, image_patch_increment[2]):
            # print(z_slice_start)
            for y_slice_start in range(0, final_y_slice_start_idx + 1, image_patch_increment[1]):
                for x_slice_start in range(0, final_x_slice_start_idx + 1, image_patch_increment[0]):
                    # print(x_slice_start, len(image_patches), image_patches[0].shape)
                    curr_image_patch = image_patches[c]
                    curr_label_patch = label_patches[c]

                    x_start_new = x_slice_start + sub_st
                    y_start_new = y_slice_start + sub_st
                    z_start_new = z_slice_start + sub_st

                    x_end = x_start_new + (sub_en - sub_st)
                    y_end = y_start_new + (sub_en - sub_st)
                    z_end = z_start_new + (sub_en - sub_st)
                    # tf.print(tf.shape(recon_image), x_start_new, y_start_new, z_start_new)
                    # tf.print(tf.shape(curr_image_patch), tf.shape(curr_label_patch))
                    recon_image[x_start_new:x_end, y_start_new:y_end, z_start_new:z_end] = curr_image_patch[
                                                                                           sub_st:sub_en,
                                                                                           sub_st:sub_en,
                                                                                           sub_st:sub_en, 0]
                    recon_label[x_start_new:x_end, y_start_new:y_end, z_start_new:z_end] = curr_label_patch[
                                                                                           sub_st:sub_en,
                                                                                           sub_st:sub_en,
                                                                                           sub_st:sub_en, 0]
                    if type(predictions) != bool:  # correct for padding during tf record generation
                        curr_pred_patch = pred_patches[c]
                        curr_logits_patch = logit_patches[c]
                        # z_start_new = z_start_new - 6
                        # z_end = z_end - 6
                        recon_pred[x_start_new:x_end, y_start_new:y_end, z_start_new:z_end] = curr_pred_patch[
                                                                                              sub_st:sub_en,
                                                                                              sub_st:sub_en,
                                                                                              sub_st:sub_en, 0]
                        recon_logits[x_start_new:x_end, y_start_new:y_end, z_start_new:z_end] = curr_logits_patch[
                                                                                                sub_st:sub_en,
                                                                                                sub_st:sub_en,
                                                                                                sub_st:sub_en, 0]

                    c += 1

        ppp = 14
        recon_image = recon_image[padding[0][0]:-padding[0][1], padding[1][0]:-padding[1][1], padding[2][0]:-padding[2][1]]
        recon_label = recon_label[padding[0][0]:-padding[0][1], padding[1][0]:-padding[1][1], padding[2][0]:-padding[2][1]]
        recon_pred = recon_pred[padding[0][0]:-padding[0][1], padding[1][0]:-padding[1][1], padding[2][0]:-padding[2][1]]
        recon_logits = recon_logits[padding[0][0]:-padding[0][1], padding[1][0]:-padding[1][1],
                       padding[2][0]:-padding[2][1]]

        assert c == n_samples, f'NUM SAMPLES MISMATCH {c} vs {n_samples}'

        img = nib.load(self.image_path)
        new_image = nib.Nifti1Image(recon_image, affine=img.affine, header=img.header)
        new_label = nib.Nifti1Image(recon_label, affine=img.affine, header=img.header)
        nib.save(new_image, self.save_path_recon_image)
        nib.save(new_label, self.save_path_recon_label)
        if type(predictions) != bool:
            new_pred = nib.Nifti1Image(recon_pred, affine=img.affine, header=img.header)
            new_logits = nib.Nifti1Image(recon_logits, affine=img.affine, header=img.header)
            nib.save(new_pred, self.save_path_recon_pred)
            nib.save(new_logits, self.save_path_recon_logits)

    def calculate_padding(self, img):
        margin = 32
        margin_halved = margin // 2
        img_shape = img.shape
        img_shape_padded = list(img.shape)
        assert len(img_shape) == 3, f'j393'

        padding = [[0, 0], [0, 0], [0, 0]]
        for dim in range(len(img_shape)):

            dim_size = img_shape[dim]
            dim_remainder = dim_size % margin
            if dim_remainder == 0:
                dim_extra = 0
            else:
                dim_extra = 32 - dim_remainder

            if dim_extra % 2 == 0:
                padl = dim_extra // 2 + margin_halved
                padr = dim_extra // 2 + margin_halved
            else:
                padl = dim_extra // 2 + margin_halved
                padr = dim_extra // 2 + 1 + margin_halved

            padding[dim][0] = padl
            padding[dim][1] = padr
            img_shape_padded[dim] += padl + padr

        if self.verbose:
            print(f"\t\t Image padded by {padding} from {img_shape} to {img_shape_padded}")
        return padding

    def load_nifti(self, path, filename):
        """    Loads a volumetric Nifti file. It returns a volume with the file's data and its metadata."""
        proxy_data = nib.load(path + filename)
        data = proxy_data.get_fdata()
        return data, proxy_data

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def generate_tfrecord(self):
        if os.path.exists(self.tfrecords_path) and os.path.exists(self.tfrecords_json_path):
            print('\t\t Using existing TFRecords and JSON')
            return False

        image_patch_dim = [64, 64, 64]  # [128, 128, 128] #
        label_patch_dim = [64, 64, 64]  # [128, 128, 128] #
        image_patch_increment = [32, 32, 32]  # [64,64,64] #[128,128,128] #          # Create one sample per 256*256*10 slice, increment z 5

        json_output = {"studies": [], "samples": 0}
        json_output["image_patch_dim"] = image_patch_dim
        json_output["label_patch_dim"] = label_patch_dim
        json_output["image_patch_increment"] = image_patch_increment

        study, metadata_study = self.load_nifti("", self.image_path)
        label = np.round(np.zeros_like(study)).astype('uint8')
        assert len(study.shape) == 3, f'Image must have exactly 3 dimensions image shape of {self.tag} is {study.shape}'

        padding = self.calculate_padding(study)
        study = np.pad(study, padding, 'constant', constant_values=(0))
        label = np.pad(label, padding, 'constant', constant_values=(0))

        study_shape = [study.shape[0], study.shape[1], study.shape[2]]
        n_x_slices = (study_shape[0] - image_patch_dim[0]) // image_patch_increment[0] + 1
        n_y_slices = (study_shape[1] - image_patch_dim[1]) // image_patch_increment[1] + 1
        n_z_slices = (study_shape[2] - image_patch_dim[2]) // image_patch_increment[2] + 1
        n_samples = n_x_slices * n_y_slices * n_z_slices

        final_x_slice_start_idx = (n_x_slices - 1) * image_patch_increment[0]
        final_y_slice_start_idx = (n_y_slices - 1) * image_patch_increment[1]
        final_z_slice_start_idx = (n_z_slices - 1) * image_patch_increment[2]
        if self.verbose:
            print(f'\t\t n_samples: {n_samples}, n_slices_each_direction: {n_x_slices, n_y_slices, n_z_slices}, final_slice_start_idx: {final_x_slice_start_idx, final_y_slice_start_idx, final_z_slice_start_idx}')

        #    Add specs into the dataset configuration file
        json_output["studies"].append({'tag': tag, 'shape': study_shape, 'padding': padding})
        json_output["samples"] = json_output["samples"] + n_samples

        # writer = tf.io.TFRecordWriter(f"C:\\raw data\\ixi mra brain all\\tf\\{tag}.tfrecords",
        # 							  tf.io.TFRecordOptions(compression_type='GZIP'))  # Create TF records file
        writer = tf.io.TFRecordWriter(self.tfrecords_path,
                                      tf.io.TFRecordOptions(compression_type='GZIP'))  # Create TF records file
        c = 0
        for z_slice_start in range(0, final_z_slice_start_idx + 1, image_patch_increment[2]):
            if self.verbose:
                print(f'\t\t {z_slice_start} / {final_z_slice_start_idx}')
            for y_slice_start in range(0, final_y_slice_start_idx + 1, image_patch_increment[1]):
                for x_slice_start in range(0, final_x_slice_start_idx + 1, image_patch_increment[0]):
                    x_slice_end = x_slice_start + image_patch_dim[0]
                    y_slice_end = y_slice_start + image_patch_dim[1]
                    z_slice_end = z_slice_start + image_patch_dim[2]
                    study_slice = study[x_slice_start:x_slice_end, y_slice_start:y_slice_end,
                                  z_slice_start: z_slice_end]
                    label_slice = label[x_slice_start:x_slice_end, y_slice_start:y_slice_end,
                                  z_slice_start: z_slice_end]

                    assert study_slice.shape == (
                    image_patch_dim[0], image_patch_dim[1], image_patch_dim[2]), f'{study_slice.shape}, {image_patch_dim}'
                    assert label_slice.shape == (
                    label_patch_dim[0], label_patch_dim[1], label_patch_dim[2]), f'{label_slice.shape}, {label_patch_dim}'

                    #    Create a feature
                    feature = {'label': self._bytes_feature(label_slice.tobytes()),
                               'image': self._bytes_feature(study_slice.tobytes())}
                    #    Create an example protocol buffer
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    #    Serialize to string and write on the file
                    writer.write(example.SerializeToString())
                    c += 1
                    writer.flush()

        assert c == n_samples, f'Number of samples saved {c} doesnt match calculated n_samples {n_samples}'
        with open(self.tfrecords_json_path, 'w') as outfile:
            json.dump(json_output, outfile)
        writer.flush()
        writer.close()
        print('\t\t New TFRecords and JSON generated')

        return True

    def run(self):
        print('\t Segmenting Images')
        print(f'\t\t Using GPU: {os.environ["CUDA_VISIBLE_DEVICES"]}')
        print(f'\t\t GPU visible in TF: {tf.config.list_physical_devices("GPU")}')

        generated_new_tfrecords = self.generate_tfrecord()
        dataset_cfg = json.load(open(self.tfrecords_json_path))
        image_size = dataset_cfg["studies"][0]["shape"]
        image_padding = dataset_cfg["studies"][0]["padding"]  # dataset_cfg["studies"][0]["padding"]
        image_patch_dim = dataset_cfg["image_patch_dim"]
        label_patch_dim = dataset_cfg["label_patch_dim"]
        image_patch_increment = dataset_cfg["image_patch_increment"]

        dataset = tf.data.TFRecordDataset(self.tfrecords_path, compression_type="GZIP")
        dataset = dataset.map(lambda x: self.parse_data(x, image_patch_dim, label_patch_dim, improper_standardisation=False,
                                                   do_intermediate_priors=False)).batch(2)

        model_test = load_model(self.segmentation_model_path, compile=False)  # compile=False added 14.9.22
        results = model_test.predict(dataset)  # DatasetGenerator(dataset, image_patch_dim, label_patch_dim, do_testing=True, do_cascade_decoupled=False)(), verbose=1)  # [2589, 256, 256, 7]
        # results = model_test(dataset)#.predict(DatasetGenerator(dataset, image_patch_dim, label_patch_dim, do_testing=True, do_cascade_decoupled=False)(), verbose=1)  # [2589, 256, 256, 7]
        self.reconstruct_patch_dataset(dataset, image_patch_dim, image_size, image_patch_increment, image_padding, predictions=results)


if __name__ == '__main__':
    start_time_global = time.perf_counter()

    with open(sys.argv[1]) as config_file:
        cfg = json.load(config_file)

    tags = cfg['tags']
    root_path = cfg['bids_root_path']
    segmentation_verbose = int(cfg['segmentation_verbose'])
    image_to_segment_suffix = cfg['image_to_segment_suffix']
    segmentation_model_path = cfg['segmentation_model_path']
    segmentation_model_threshold = float(cfg['segmentation_model_threshold'])
    images_path = f"{root_path}/rawdata"
    segmentations_path = f"{root_path}/derivatives/MRAtoBG/segmentations"
    n_subjects = len(tags)
    gpu_selection = cfg['gpu_selection']

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_selection
    if not os.path.exists(images_path):
        raise ValueError(f'There must be a folder at {images_path}')
    if not os.path.exists(segmentations_path):
        os.mkdir(segmentations_path)
        print(f'Directory created for Segmentations')

    # Before any processing ensure all expected files exist
    for tag in tags:
        image_to_segment_fpath = f'{images_path}/{tag}/anat/{tag}_{image_to_segment_suffix}.nii.gz'
        assert os.path.exists(image_to_segment_fpath), f"This file does not exist {image_to_segment_fpath}"

    for subj_ix, tag in enumerate(tags):
        print(f'Processing started for subject {subj_ix + 1} of {n_subjects}: {tag}')
        segmentation_start_time = time.perf_counter()
        Segmentation_obj = Segmentation(subj_ix, tag, images_path, segmentations_path, image_to_segment_suffix, segmentation_model_path, segmentation_model_threshold, segmentation_verbose)
        Segmentation_obj.run()
        curr_segmentation_minutes = (time.perf_counter() - segmentation_start_time) / 60
        print(f'\t\t Segmentation completed in {curr_segmentation_minutes:.2f} minutes')

    global_minutes = (time.perf_counter() - start_time_global) / 60
    print(f'Segmentation Module processed {n_subjects} subjects in {global_minutes:.2f} minutes')
