import torch
import torch.nn.functional as F

from ._base import Distiller


class AvgFeat(Distiller):
    """Feature-level distillation with weighted averaging over multiple teachers."""

    def __init__(self, student, teacher, cfg):
        super().__init__(student, teacher, cfg.DISTILLER.TEACHER_WEIGHTS)
        self.ce_loss_weight = cfg.AVGFEAT.LOSS.CE_WEIGHT
        self.feat_loss_weight = cfg.AVGFEAT.LOSS.FEAT_WEIGHT
        self.preact_loss_weight = cfg.AVGFEAT.LOSS.PREACT_WEIGHT
        self.pooled_loss_weight = cfg.AVGFEAT.LOSS.POOLED_WEIGHT

    def forward_train(self, image, target, **kwargs):
        logits_student, feature_student = self.student(image)

        with torch.no_grad():
            teacher_features = []
            for teacher in self.teachers:
                _, feat = teacher(image)
                teacher_features.append(feat)

        aggregated_features = self._aggregate_teacher_features(teacher_features)

        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        total_feat_loss = logits_student.new_tensor(0.0)

        if "feats" in feature_student and "feats" in aggregated_features:
            total_feat_loss = total_feat_loss + self._compute_feature_loss(
                feature_student["feats"], aggregated_features["feats"]
            )

        if (
            self.preact_loss_weight > 0
            and "preact_feats" in feature_student
            and "preact_feats" in aggregated_features
        ):
            total_feat_loss = total_feat_loss + self.preact_loss_weight * self._compute_feature_loss(
                feature_student["preact_feats"], aggregated_features["preact_feats"]
            )

        if (
            self.pooled_loss_weight > 0
            and "pooled_feat" in feature_student
            and "pooled_feat" in aggregated_features
        ):
            pooled_teacher = aggregated_features["pooled_feat"]
            pooled_student = feature_student["pooled_feat"]
            pooled_teacher = self._match_spatial_shape(pooled_teacher, pooled_student)
            total_feat_loss = total_feat_loss + self.pooled_loss_weight * F.mse_loss(
                pooled_student, pooled_teacher
            )

        loss_feat = self.feat_loss_weight * total_feat_loss

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_feat": loss_feat,
        }
        return logits_student, losses_dict

    def _aggregate_teacher_features(self, teacher_features):
        aggregated = None
        weights = self.teacher_weights
        for idx, feats in enumerate(teacher_features):
            weight = float(weights[idx].item())
            if aggregated is None:
                aggregated = self._scale_structure(feats, weight)
            else:
                aggregated = self._add_scaled_structure(aggregated, feats, weight)
        return aggregated

    def _scale_structure(self, data, weight):
        if torch.is_tensor(data):
            return data * weight
        if isinstance(data, dict):
            return {k: self._scale_structure(v, weight) for k, v in data.items()}
        if isinstance(data, list):
            return [self._scale_structure(item, weight) for item in data]
        if isinstance(data, tuple):
            return tuple(self._scale_structure(item, weight) for item in data)
        return data

    def _add_scaled_structure(self, accum, data, weight):
        if torch.is_tensor(data):
            return accum + data * weight
        if isinstance(data, dict):
            return {
                k: self._add_scaled_structure(accum[k], data[k], weight)
                for k in data
            }
        if isinstance(data, list):
            return [
                self._add_scaled_structure(a, b, weight)
                for a, b in zip(accum, data)
            ]
        if isinstance(data, tuple):
            return tuple(
                self._add_scaled_structure(a, b, weight)
                for a, b in zip(accum, data)
            )
        return accum

    def _compute_feature_loss(self, student_feats, teacher_feats):
        student_seq = self._ensure_sequence(student_feats)
        teacher_seq = self._ensure_sequence(teacher_feats)
        if len(student_seq) != len(teacher_seq):
            raise ValueError(
                "Student and teacher feature lists must have the same length."
            )
        losses = []
        for f_s, f_t in zip(student_seq, teacher_seq):
            f_t = self._match_spatial_shape(f_t, f_s)
            losses.append(F.mse_loss(f_s, f_t))
        if not losses:
            return self.teacher_weights.new_tensor(0.0)
        return sum(losses) / len(losses)

    def _ensure_sequence(self, value):
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, list):
            return value
        return [value]

    def _match_spatial_shape(self, source, target):
        if not torch.is_tensor(source) or not torch.is_tensor(target):
            return source
        if source.dim() >= 4 and target.dim() >= 4 and source.shape[2:] != target.shape[2:]:
            source = F.adaptive_avg_pool2d(source, target.shape[2:])
        return source
