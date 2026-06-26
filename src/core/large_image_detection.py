"""
大图检测包装器

在检测前自动判断是否需要切割，处理超长漫画图片
每个切片独立检测，然后合并结果
"""

import logging

import cv2
import numpy as np
from PIL import Image

from src.core.detector.aux_yolo import maybe_merge_with_aux_yolo
from src.core.detector.data_types import TextBlock, TextLine, DetectionResult
from src.core.detector.base import BaseTextDetector
from src.core.detector.textline_merge import merge_textlines
from src.core.detector.postprocess import postprocess_blocks
from src.utils.image_rearrange import (
    check_needs_rearrange,
    slice_image_for_detection,
    transform_textlines_to_original,
    merge_masks_from_patches,
    DEFAULT_TARGET_SIZE
)

logger = logging.getLogger("LargeImageDetection")


def _aabb_intersection_area(first: TextLine, second: TextLine) -> float:
    ax1, ay1, ax2, ay2 = first.xyxy
    bx1, by1, bx2, by2 = second.xyxy
    width = max(0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0, min(ay2, by2) - max(ay1, by1))
    return float(width * height)


def _intersection_area(first: TextLine, second: TextLine) -> float:
    try:
        return float(first.polygon.intersection(second.polygon).area)
    except Exception:
        return _aabb_intersection_area(first, second)


def _is_preferred_textline(
    candidate_index: int,
    candidate: TextLine,
    existing_index: int,
    existing: TextLine,
) -> bool:
    candidate_score = (candidate.confidence, candidate.area, -candidate_index)
    existing_score = (existing.confidence, existing.area, -existing_index)
    return candidate_score > existing_score


def _is_duplicate_textline(
    first: TextLine,
    second: TextLine,
    overlap_threshold: float = 0.8,
    containment_threshold: float = 0.9,
    center_distance_ratio: float = 0.5,
) -> bool:
    if first.direction != second.direction:
        return False

    first_area = first.area
    second_area = second.area
    if first_area <= 0 or second_area <= 0:
        return False

    intersection = _intersection_area(first, second)
    if intersection <= 0:
        return False

    union = first_area + second_area - intersection
    iou = intersection / union if union > 0 else 0.0
    smaller_coverage = intersection / min(first_area, second_area)
    area_ratio = min(first_area, second_area) / max(first_area, second_area)

    first_w, first_h = first.xywh[2:]
    second_w, second_h = second.xywh[2:]
    center_limit = max(
        3.0,
        min(first_w, first_h, second_w, second_h) * center_distance_ratio,
    )
    centers_close = np.linalg.norm(first.center - second.center) <= center_limit

    if iou >= 0.75:
        return True
    if smaller_coverage >= overlap_threshold and centers_close:
        return True
    return (
        smaller_coverage >= containment_threshold
        and area_ratio >= 0.65
        and centers_close
    )


def _deduplicate_overlapping_textlines(textlines: list[TextLine]) -> list[TextLine]:
    if len(textlines) < 2:
        return textlines

    indexed_lines = list(enumerate(textlines))
    candidates = sorted(
        indexed_lines,
        key=lambda item: (item[1].confidence, item[1].area, -item[0]),
        reverse=True,
    )
    kept: list[tuple[int, TextLine]] = []

    for candidate_index, candidate in candidates:
        duplicate_at = None
        for kept_index, (_, kept_line) in enumerate(kept):
            if _is_duplicate_textline(candidate, kept_line):
                duplicate_at = kept_index
                break

        if duplicate_at is None:
            kept.append((candidate_index, candidate))
            continue

        existing_index, existing_line = kept[duplicate_at]
        output_index = min(candidate_index, existing_index)
        if _is_preferred_textline(
            candidate_index, candidate, existing_index, existing_line
        ):
            kept[duplicate_at] = (output_index, candidate)
        else:
            kept[duplicate_at] = (output_index, existing_line)

    kept.sort(key=lambda item: item[0])
    return [line for _, line in kept]


class LargeImageDetectorWrapper:
    """
    大图检测器包装器
    
    封装现有检测器，自动处理超长图片的切割和拼接
    """
    
    def __init__(
        self,
        detector: BaseTextDetector,
        target_size: int = DEFAULT_TARGET_SIZE
    ):
        self.detector = detector
        self.target_size = target_size
    
    def detect(
        self,
        image: Image.Image,
        merge_lines: bool = None,
        edge_ratio_threshold: float = 0.0,
        expand_ratio: float = 0,
        expand_top: float = 0,
        expand_bottom: float = 0,
        expand_left: float = 0,
        expand_right: float = 0,
        **kwargs
    ) -> DetectionResult:
        """带自动切割的检测"""
        img_np = np.array(image.convert('RGB'))
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        im_w, im_h = image.width, image.height
        
        needs_rearrange, _ = check_needs_rearrange(img_cv, self.target_size)
        
        if not needs_rearrange:
            logger.debug("图像尺寸正常，直接检测")
            return self.detector.detect(
                image,
                merge_lines=merge_lines,
                edge_ratio_threshold=edge_ratio_threshold,
                expand_ratio=expand_ratio,
                expand_top=expand_top,
                expand_bottom=expand_bottom,
                expand_left=expand_left,
                expand_right=expand_right,
                **kwargs
            )
        
        logger.info(f"图像尺寸过大 ({im_w}x{im_h})，启用切割检测")
        
        return self._detect_with_slicing(
            img_cv, im_w, im_h,
            merge_lines=merge_lines if merge_lines is not None else self.detector.requires_merge,
            edge_ratio_threshold=edge_ratio_threshold,
            expand_ratio=expand_ratio,
            expand_top=expand_top,
            expand_bottom=expand_bottom,
            expand_left=expand_left,
            expand_right=expand_right,
            **kwargs
        )
    
    def _detect_with_slicing(
        self,
        img_cv: np.ndarray,
        im_w: int,
        im_h: int,
        merge_lines: bool,
        edge_ratio_threshold: float,
        expand_ratio: float,
        expand_top: float,
        expand_bottom: float,
        expand_left: float,
        expand_right: float,
        **kwargs
    ) -> DetectionResult:
        """执行切割检测"""
        
        # 1. 切割图像
        patches, context = slice_image_for_detection(
            img_cv,
            tgt_size=self.target_size,
            verbose=True
        )
        
        if not patches or not context.is_rearranged:
            logger.warning("切割失败，回退到普通检测")
            img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            return self.detector.detect(
                img_pil,
                merge_lines=merge_lines,
                edge_ratio_threshold=edge_ratio_threshold,
                expand_ratio=expand_ratio,
                expand_top=expand_top,
                expand_bottom=expand_bottom,
                expand_left=expand_left,
                expand_right=expand_right,
                **kwargs
            )
        
        # 2. 逐切片检测
        all_textlines = []
        all_masks = []
        enable_aux_yolo_detection = kwargs.pop('enable_aux_yolo_detection', None)
        aux_yolo_conf_threshold = kwargs.pop('aux_yolo_conf_threshold', None)
        aux_yolo_overlap_threshold = kwargs.pop('aux_yolo_overlap_threshold', None)

        if enable_aux_yolo_detection is None:
            from src.shared import constants

            enable_aux_yolo_detection = constants.ENABLE_AUX_YOLO_DETECTION
        
        logger.info(f"开始检测 {len(patches)} 个切片...")
        
        for patch_idx, patch in enumerate(patches):
            logger.info(f"检测切片 {patch_idx + 1}/{len(patches)}...")
            
            patch_textlines, patch_mask = self.detector._detect_raw(patch, **kwargs)
            if enable_aux_yolo_detection:
                patch_textlines = maybe_merge_with_aux_yolo(
                    patch,
                    patch_textlines,
                    detector_type=getattr(self.detector, 'detector_id', ''),
                    enabled=enable_aux_yolo_detection,
                    conf_threshold=aux_yolo_conf_threshold,
                    overlap_threshold=aux_yolo_overlap_threshold,
                )
            
            num_lines = len(patch_textlines) if patch_textlines else 0
            logger.info(f"  切片 {patch_idx + 1}: 检测到 {num_lines} 个文本行")
            
            if patch_textlines:
                transformed_textlines = transform_textlines_to_original(
                    patch_textlines, patch_idx, context
                )
                all_textlines.extend(transformed_textlines)
                logger.info(f"  切片 {patch_idx + 1}: 坐标转换后 {len(transformed_textlines)} 个文本行")
            
            if patch_mask is not None:
                all_masks.append(patch_mask)
        
        logger.info(f"切割检测完成: 共检测到 {len(all_textlines)} 个文本行 (来自 {len(patches)} 个切片)")
        
        # 3. 合并掩码
        final_mask = None
        if all_masks:
            try:
                final_mask = merge_masks_from_patches(all_masks, context)
            except Exception as e:
                logger.warning(f"掩码合并失败: {e}")
        
        # 4. 处理文本行
        if not all_textlines:
            logger.info("未检测到文本区域")
            return DetectionResult(blocks=[], mask=final_mask, raw_lines=[])
        
        # 剪裁到图像边界
        for line in all_textlines:
            line.clip(im_w, im_h)
        
        # 过滤无效文本行
        valid_textlines = [
            line for line in all_textlines
            if line.area > 16 and all(0 <= pt[0] <= im_w and 0 <= pt[1] <= im_h for pt in line.pts)
        ]
        
        logger.info(f"有效文本行: {len(valid_textlines)} / {len(all_textlines)}")

        deduplicated_textlines = _deduplicate_overlapping_textlines(valid_textlines)
        if len(deduplicated_textlines) != len(valid_textlines):
            logger.info(
                "Removed %d duplicate textlines from overlapping slices",
                len(valid_textlines) - len(deduplicated_textlines),
            )
        valid_textlines = deduplicated_textlines
        
        # 5. 合并文本行
        if merge_lines and valid_textlines:
            blocks = merge_textlines(
                valid_textlines, im_w, im_h,
                edge_ratio_threshold=edge_ratio_threshold,
                verbose=True
            )
            logger.info(f"合并后得到 {len(blocks)} 个文本块")
        else:
            blocks = [TextBlock(lines=[line]) for line in valid_textlines]
        
        # 6. 后处理
        blocks = postprocess_blocks(
            blocks, im_w, im_h,
            expand_ratio=expand_ratio,
            expand_top=expand_top,
            expand_bottom=expand_bottom,
            expand_left=expand_left,
            expand_right=expand_right
        )
        
        return DetectionResult(
            blocks=blocks,
            mask=final_mask,
            raw_lines=valid_textlines
        )
