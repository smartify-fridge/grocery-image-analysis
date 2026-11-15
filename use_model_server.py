"""
Single-image version of the use_model.py pipeline for server.py.
Reuses use_model's steps and defaults but writes no files.
"""
from types import SimpleNamespace
from mobilenet import MobileNet
from use_model import (
    CLASSIFIER_CLASSES,
    CLASSIFIER_HEAD_RELU,
    CLASSIFIER_WEIGHTS,
    CLASS_BATCH_SIZE,
    CLASSIFIER_CONFIDENCE_THRESHOLD,
    CONTAINMENT_THRESHOLD,
    BOX_NMS_THRESHOLD,
    DETECTOR_CLASSES,
    DINO_BOX_THRESHOLD,
    DINO_TEXT_THRESHOLD,
    GROUP_BOX_MIN_ITEMS,
    MASK_NMS_THRESHOLD,
    MAX_MASK_AREA_RATIO,
    MIN_MASK_AREA_RATIO,
    OUTPUT_DIR,
    PREFILTER_IOU_THRESHOLD,
    SAM2_SCORE_THRESHOLD,
    classify_detections,
    detect_with_dino,
    load_class_mapping,
    load_classifier,
    load_grounding_dino,
    load_sam2,
    make_prompt_chunks,
    remove_duplicate_predictions,
    remove_group_boxes,
    remove_near_identical_boxes,
    segment_with_sam2,
)

# Same defaults as the use_model.py command line flags
DEFAULT_SETTINGS = {
    "detector_classes": DETECTOR_CLASSES,
    "classifier_classes": CLASSIFIER_CLASSES,
    "weights": CLASSIFIER_WEIGHTS,
    "head_relu": CLASSIFIER_HEAD_RELU,
    "box_threshold": DINO_BOX_THRESHOLD,
    "text_threshold": DINO_TEXT_THRESHOLD,
    "sam_threshold": SAM2_SCORE_THRESHOLD,
    "classifier_threshold": CLASSIFIER_CONFIDENCE_THRESHOLD,
    "prefilter_iou": PREFILTER_IOU_THRESHOLD,
    "box_nms": BOX_NMS_THRESHOLD,
    "mask_nms": MASK_NMS_THRESHOLD,
    "containment": CONTAINMENT_THRESHOLD,
    "keep_duplicates": False,
    "min_area": MIN_MASK_AREA_RATIO,
    "max_area": MAX_MASK_AREA_RATIO,
    "batch_size": CLASS_BATCH_SIZE,
    "crop_padding": 0.05,
    "crop_mode": "box",
}

def make_settings(**overrides):
    unknown = set(overrides) - set(DEFAULT_SETTINGS)
    if unknown:
        raise ValueError(f"Unknown settings: {', '.join(sorted(unknown))}")
    return SimpleNamespace(**{**DEFAULT_SETTINGS, **overrides})

def load_models(settings):
    """Load every model once. Returns the model arguments for analyze_image."""
    detector_classes = load_class_mapping(settings.detector_classes)

    if settings.head_relu:
        MobileNet.fix_metal_relu()

    classifier, label_map = load_classifier(
        settings.weights,
        settings.classifier_classes,
        settings.head_relu
    )
    processor, detector = load_grounding_dino()
    predictor = load_sam2()
    prompt_chunks = make_prompt_chunks(detector_classes, settings.batch_size)

    return {
        "processor": processor,
        "detector": detector,
        "prompt_chunks": prompt_chunks,
        "predictor": predictor,
        "classifier": classifier,
        "label_map": label_map
    }

def analyze_image(
    image,
    settings,
    processor,
    detector,
    prompt_chunks,
    predictor,
    classifier,
    label_map
):
    """
    Same steps as use_model.process_image on one PIL image, without saving
    crops, annotated images or CSVs. Returns the kept predictions, most
    confident first.
    """
    detections = detect_with_dino(
        image,
        processor,
        detector,
        prompt_chunks,
        settings.box_threshold,
        settings.text_threshold
    )

    if not settings.keep_duplicates:
        detections = remove_near_identical_boxes(detections, settings.prefilter_iou)
        detections = remove_group_boxes(
            detections,
            settings.containment,
            settings.box_nms,
            GROUP_BOX_MIN_ITEMS
        )

    segmented = segment_with_sam2(
        image,
        detections,
        predictor,
        settings.sam_threshold,
        settings.min_area,
        settings.max_area
    )
    # output_folder and image_stem only name crop paths; nothing is written
    predictions = classify_detections(
        image=image,
        detections=segmented,
        classifier=classifier,
        label_map=label_map,
        output_folder=OUTPUT_DIR,
        image_stem="request",
        confidence_threshold=settings.classifier_threshold,
        crop_padding=settings.crop_padding,
        crop_mode=settings.crop_mode
    )

    if settings.keep_duplicates:
        return sorted(
            (p for p in predictions if p["accepted"]),
            key=lambda x: x["classifier_confidence"],
            reverse=True
        )

    return remove_duplicate_predictions(
        predictions,
        settings.box_nms,
        settings.mask_nms,
        settings.containment
    )
