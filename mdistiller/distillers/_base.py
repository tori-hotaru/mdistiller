import torch
import torch.nn as nn
import torch.nn.functional as F


class Distiller(nn.Module):
    def __init__(self, student, teacher, teacher_weights=None):
        super(Distiller, self).__init__()
        self.student = student

        if isinstance(teacher, nn.ModuleList):
            teacher_modules = list(teacher)
        elif isinstance(teacher, (list, tuple)):
            teacher_modules = list(teacher)
        else:
            teacher_modules = [teacher]

        if len(teacher_modules) == 0:
            raise ValueError("At least one teacher network must be provided.")

        self.teachers = nn.ModuleList(teacher_modules)
        # keep the original attribute name for backward compatibility
        self.teacher = self.teachers[0]
        self._num_teachers = len(self.teachers)

        if teacher_weights is None or len(teacher_weights) == 0:
            processed_weights = [1.0] * self._num_teachers
        else:
            processed_weights = [float(w) for w in list(teacher_weights)]

        if len(processed_weights) != self._num_teachers:
            raise ValueError(
                "The number of teacher weights must match the number of teachers."
            )

        weight_sum = sum(processed_weights)
        if weight_sum <= 0:
            raise ValueError("The sum of teacher weights must be positive.")

        normalized_weights = [w / weight_sum for w in processed_weights]
        self.register_buffer(
            "_teacher_weights", torch.tensor(normalized_weights, dtype=torch.float32)
        )

    def train(self, mode=True):
        # teacher as eval mode by default
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        for teacher in self.teachers:
            teacher.eval()
        return self

    @property
    def teacher_weights(self):
        return self._teacher_weights

    @property
    def num_teachers(self):
        return self._num_teachers

    def get_learnable_parameters(self):
        # if the method introduces extra parameters, re-impl this function
        return [v for k, v in self.student.named_parameters()]

    def get_extra_parameters(self):
        # calculate the extra parameters introduced by the distiller
        return 0

    def forward_train(self, **kwargs):
        # training function for the distillation method
        raise NotImplementedError()

    def forward_test(self, image):
        return self.student(image)[0]

    def forward(self, **kwargs):
        if self.training:
            return self.forward_train(**kwargs)
        return self.forward_test(kwargs["image"])


class Vanilla(nn.Module):
    def __init__(self, student):
        super(Vanilla, self).__init__()
        self.student = student

    def get_learnable_parameters(self):
        return [v for k, v in self.student.named_parameters()]

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        loss = F.cross_entropy(logits_student, target)
        return logits_student, {"ce": loss}

    def forward(self, **kwargs):
        if self.training:
            return self.forward_train(**kwargs)
        return self.forward_test(kwargs["image"])

    def forward_test(self, image):
        return self.student(image)[0]
