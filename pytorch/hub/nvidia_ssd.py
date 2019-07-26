import matplotlib.patches as patches
from matplotlib import pyplot as plt
import torch

precision = 'fp32'
ssd_model = torch.hub.load(
    'NVIDIA/DeepLearningExamples', 'nvidia_ssd', model_math=precision)

utils = torch.hub.load('NVIDIA/DeepLearningExamples',
                       'nvidia_ssd_processing_utils')

ssd_model.to('cuda')
ssd_model.eval()


uris = ['./1.jpg', './2.jpg', './3.jpg', './dog.jpg']
inputs = [utils.prepare_input(uri) for uri in uris]

tensor = utils.prepare_tensor(inputs, precision == 'fp16')


with torch.no_grad():
    detections_batch = ssd_model(tensor)

results_per_input = utils.decode_results(detections_batch)
best_results_per_input = [utils.pick_best(
    results, 0.40) for results in results_per_input]

#classes_to_labels = utils.get_coco_object_dictionary()
classes_to_labels = ['person',
                     'bicycle',
                     'car',
                     'motorcycle',
                     'airplane',
                     'bus',
                     'train',
                     'truck',
                     'boat',
                     'traffic light',
                     'fire hydrant',
                     'stop sign',
                     'parking meter',
                     'bench',
                     'bird',
                     'cat',
                     'dog',
                     'horse',
                     'sheep',
                     'cow',
                     'elephant',
                     'bear',
                     'zebra',
                     'giraffe',
                     'backpack',
                     'umbrella',
                     'handbag',
                     'tie',
                     'suitcase',
                     'frisbee',
                     'skis',
                     'snowboard',
                     'sports ball',
                     'kite',
                     'baseball bat',
                     'baseball glove',
                     'skateboard',
                     'surfboard',
                     'tennis racket',
                     'bottle',
                     'wine glass',
                     'cup',
                     'fork',
                     'knife',
                     'spoon',
                     'bowl',
                     'banana',
                     'apple',
                     'sandwich',
                     'orange',
                     'broccoli',
                     'carrot',
                     'hot dog',
                     'pizza',
                     'donut',
                     'cake',
                     'chair',
                     'couch',
                     'potted plant',
                     'bed',
                     'dining table',
                     'toilet',
                     'tv',
                     'laptop',
                     'mouse',
                     'remote',
                     'keyboard',
                     'cell phone',
                     'microwave',
                     'oven',
                     'toaster',
                     'sink',
                     'refrigerator',
                     'book',
                     'clock',
                     'vase',
                     'scissors',
                     'teddy bear',
                     'hair drier',
                     'toothbrush']


for image_idx in range(len(best_results_per_input)):
    fig, ax = plt.subplots(1)
    # Show original, denormalized image...
    image = inputs[image_idx] / 2 + 0.5
    ax.imshow(image)
    # ...with detections
    bboxes, classes, confidences = best_results_per_input[image_idx]
    for idx in range(len(bboxes)):
        left, bot, right, top = bboxes[idx]
        x, y, w, h = [val * 300 for val in [left,
                                            bot, right - left, top - bot]]
        rect = patches.Rectangle(
            (x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y, "{} {:.0f}%".format(
            classes_to_labels[classes[idx] - 1], confidences[idx]*100), bbox=dict(facecolor='white', alpha=0.5))
    plt.show()
