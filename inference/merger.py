import copy
import itertools
import numpy as np
from utils import calc_iou, to_xywh, to_xyxy, to_dict


class Merger(object):

    def __init__(self, overlap_pixels=0):
        self.overlap_pixels = overlap_pixels

    def merge(self, inference_results_, all_split_images_):
        """
        Postprocesses the images. This includes:
        - Merging the bounding boxes of the split images into the original image
        - Deleting the preprocessed images
        :param results_: results from the inference
        :return: merged results
        """
        # TODO Split images is problematic, because the ids are not the same as the original ones

        keys_to_add = []
        last_net = len(inference_results_) - 1
        results = copy.deepcopy(inference_results_)
        merged_results = [[] for _ in range(len(results))]

        for net_num, result in enumerate(results):
            split_images_ = copy.deepcopy(all_split_images_)

            for key in split_images_:
                ids = list(split_images_[key].keys())
                # get bounding boxes of the pairs
                bboxes = [[] for _ in ids]
                labels = []
                scores = []
                for i, id in enumerate(ids):
                    img_results = result[id]
                    for img_result in img_results:
                        bbox = to_xywh(img_result['box'])
                        #bbox = [val for val in img_result['box'].values()]
                        bboxes[i].append(bbox)
                        labels.append(img_result['label'])
                        scores.append(img_result['score'])
                # merge the results of the split images
                merged_bbox = self.merge_images(key, bboxes, ids, split_images_)
                # add the merged results to the new results
                picture_results = []
                for r in range(len(merged_bbox)):
                    picture_results.append(to_dict(merged_bbox[r], labels[r], scores[r]))
                merged_results[net_num].append(picture_results)
                if net_num == last_net:
                    keys_to_add.append(key)

        return merged_results, keys_to_add

    def merge_images(self, key, bboxes, ids, split_images_):
        """
        Recursively merges the results of the split images into the original image.
        :param key: key of the original image
        :param bboxes: results from the inference
        :param ids: ids
        :return: merged results
        """

        # split ids into pairs of four
        _ids = [i for i in range(len(ids))]
        id_pairs = [_ids[x:x+4] for x in range(0, len(_ids), 4)] #[[0,1,2,3],[4,5,6,7], ...]
        pairs = [ids[x:x+4] for x in range(0, len(ids), 4)] # [[16,17,18,19],[20,21,22,23], ...]

        while len(pairs[0]) > 1:
            new_pairs = []
            new_bboxes = []
            collapse = []
            for i, pair in enumerate(pairs):
                pairboxes = [bboxes[j] for j in id_pairs[i]]
                _bboxes = self.merge_results_of_four(key, pairboxes, pair, split_images_)
                new_bboxes.append(_bboxes)
                new_pairs.append(i)
                collapse.append(pair)
            pairs = [new_pairs[x:x+4] for x in range(0, len(new_pairs), 4)]
            bboxes = new_bboxes
            for i, pair in enumerate(collapse):
                new_height = split_images_[key][pair[0]][0] + split_images_[key][pair[2]][0]
                new_width = split_images_[key][pair[0]][1] + split_images_[key][pair[1]][1]
                for j in pair:
                    split_images_[key].pop(j)
                split_images_[key][i] = (new_height, new_width)
        return bboxes[0]

    def merge_results_of_four(self, key, bboxes, pair, split_images_):
        """
        Merges the results of four images into one.
        :param bboxes: results from the inference
        :param pair: pair to merge
        :return: merged results
        """

        sizes = [split_images_[key][i] for i in pair]
        w_offset = sizes[0][1]
        h_offset = sizes[0][0]

        # Modify bboxes[1] aka top right
        for bbox in bboxes[1]:
            bbox[0] += w_offset
        #bboxes[1][:, [0]] += w_offset

        # # Modify bboxes[2] aka bottom left
        for bbox in bboxes[2]:
            bbox[1] += h_offset
        #bboxes[2][:, [1]] += h_offset

        # # Modify bboxes[1] aka bottom right
        for bbox in bboxes[3]:
            bbox[0] += w_offset
            bbox[1] += h_offset
        # bboxes[3][:, [0]] += w_offset
        # bboxes[3][:, [1]] += h_offset

        # # Concatenate bboxes
        # new_bboxes = np.concatenate((bboxes[0], bboxes[1], bboxes[2], bboxes[3]), axis=0)
        # Flatten the list
        new_bboxes = list(itertools.chain.from_iterable(bboxes))

        return new_bboxes

    def merge_bboxes(self, results):
        """
        Calculates the IoU for all results and returns new bounding boxes where the bounding boxes match
        :param results: results bounding boxes to calculate the IoU for
        :return: new matching bounding boxes
        """
        net_cnt = len(results)
        net_ids = list(range(net_cnt))
        num_images = len(results[0])
        map = {'fire': 0, 'vehicle': 1, 'human': 2}
        num_classes = 3 # TODO get number of classes
        new_results = [[] for _ in range(num_images)]
        if net_cnt == 1:
            return results[0]
        # Iterate over all images
        for img_idx in range(num_images):
            # Get every bbox for this image
            bboxes = [[to_xywh(instance['box']) for instance in results[net_idx][img_idx]] for net_idx in range(net_cnt)]
            labels = [[instance['label'] for instance in results[net_idx][img_idx]] for net_idx in range(net_cnt)]
            scores = [[instance['score'] for instance in results[net_idx][img_idx]] for net_idx in range(net_cnt)]

            # Organize bounding boxes by class
            class_bboxes = [[[] for _ in range(len(results))] for _ in range(num_classes)]
            class_scores = [[[] for _ in range(len(results))] for _ in range(num_classes)]
            for net_idx, net_bboxes in enumerate(bboxes):
                for instance_idx, bbox in enumerate(net_bboxes):
                    class_idx = map[labels[net_idx][instance_idx]]  # Assuming label corresponds to class index
                    score = scores[net_idx][instance_idx]
                    class_bboxes[class_idx][net_idx].append(bbox)
                    class_scores[class_idx][net_idx].append(score)
            # Order by Class for saving computation
            new_bboxes = [[] for _ in range(num_classes)]

            # Find consensus bounding boxes
            for class_idx in range(num_classes):
                for net_a, net_b in itertools.combinations(net_ids, 2):
                    for i, bbox_a in enumerate(class_bboxes[class_idx][net_a]):
                        for j, bbox_b in enumerate(class_bboxes[class_idx][net_b]):
                            iou = calc_iou(bbox_a, bbox_b)
                            if iou > 0.3:
                                bbox = (np.array(bbox_a) + np.array(bbox_b)) / 2
                                score_a = class_scores[class_idx][net_a][i]
                                score_b = class_scores[class_idx][net_b][j]
                                score = (score_a + score_b) / 2
                                found, _idx = self.found_previously(bbox_a, new_bboxes[class_idx])
                                if not found:
                                    new_bbox = {'box': to_xyxy(bbox), 'label': list(map.keys())[list(map.values()).index(class_idx)], 'score': score}
                                    new_bboxes[class_idx].append(new_bbox)
            # Flatten again
            new_bboxes = list(itertools.chain.from_iterable(new_bboxes))
            new_results[img_idx] = new_bboxes
        return new_results

    def found_previously(self, bbox, bboxes):
        for idx, existing_bbox in enumerate(bboxes):
            iou = calc_iou(bbox, to_xywh(existing_bbox['box']))
            if iou > 0.05:
                return True, idx
        return False, None
