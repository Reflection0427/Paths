from abc import ABC
import collections

import torch
import torch.nn.functional as F
from torch import autograd, nn


class CM_Mix_Interactive(autograd.Function):
    @staticmethod
    def forward(ctx, inputs_rgb, inputs_event, indexes, features_rgb, features_event, momentum, alpha=0.1):
        ctx.features_rgb = features_rgb
        ctx.features_event = features_event
        ctx.momentum = momentum
        ctx.alpha = alpha
        ctx.save_for_backward(inputs_rgb, inputs_event, indexes)

        outputs_rgb = inputs_rgb.mm(ctx.features_rgb.to(inputs_rgb.dtype).t())
        outputs_event = inputs_event.mm(ctx.features_event.to(inputs_event.dtype).t())
        return outputs_rgb, outputs_event

    @staticmethod
    def backward(ctx, grad_outputs_rgb, grad_outputs_event):
        inputs_rgb, inputs_event, indexes = ctx.saved_tensors
        half_dim_rgb = len(ctx.features_rgb) // 2
        half_dim_event = len(ctx.features_event) // 2

        grad_inputs_rgb = None
        grad_inputs_event = None
        if ctx.needs_input_grad[0]:
            grad_inputs_rgb = grad_outputs_rgb.mm(ctx.features_rgb.to(grad_outputs_rgb.dtype))
        if ctx.needs_input_grad[1]:
            grad_inputs_event = grad_outputs_event.mm(ctx.features_event.to(grad_outputs_event.dtype))

        batch_centers_rgb = collections.defaultdict(list)
        batch_centers_event = collections.defaultdict(list)
        for feat_rgb, feat_event, index in zip(inputs_rgb, inputs_event, indexes.tolist()):
            batch_centers_rgb[index].append(feat_rgb)
            batch_centers_event[index].append(feat_event)

        for index, feat_list_rgb in batch_centers_rgb.items():
            feat_list_event = batch_centers_event[index]
            feats_rgb = torch.stack(feat_list_rgb, dim=0)
            feats_event = torch.stack(feat_list_event, dim=0)

            mean_rgb = feats_rgb.mean(dim=0)
            mean_event = feats_event.mean(dim=0)

            old_center_rgb = ctx.features_rgb[index].clone()
            old_center_event = ctx.features_event[index].clone()

            injected_rgb = F.normalize(mean_rgb + ctx.alpha * mean_event, dim=0)
            injected_event = F.normalize(mean_event + ctx.alpha * mean_rgb, dim=0)

            ctx.features_rgb[index] = ctx.features_rgb[index] * ctx.momentum + (1 - ctx.momentum) * injected_rgb.to(ctx.features_rgb.dtype)
            ctx.features_rgb[index] = F.normalize(ctx.features_rgb[index], dim=0)
            ctx.features_event[index] = ctx.features_event[index] * ctx.momentum + (1 - ctx.momentum) * injected_event.to(ctx.features_event.dtype)
            ctx.features_event[index] = F.normalize(ctx.features_event[index], dim=0)

            target_center_rgb = old_center_rgb.to(feats_rgb.dtype).unsqueeze(0)
            sims_rgb = torch.mm(feats_rgb, target_center_rgb.t()).squeeze(1)
            hard_feat_rgb = feats_rgb[torch.argmin(sims_rgb)]
            ctx.features_rgb[index + half_dim_rgb] = ctx.features_rgb[index + half_dim_rgb] * ctx.momentum + (1 - ctx.momentum) * hard_feat_rgb.to(ctx.features_rgb.dtype)
            ctx.features_rgb[index + half_dim_rgb] = F.normalize(ctx.features_rgb[index + half_dim_rgb], dim=0)

            target_center_event = old_center_event.to(feats_event.dtype).unsqueeze(0)
            sims_event = torch.mm(feats_event, target_center_event.t()).squeeze(1)
            hard_feat_event = feats_event[torch.argmin(sims_event)]
            ctx.features_event[index + half_dim_event] = ctx.features_event[index + half_dim_event] * ctx.momentum + (1 - ctx.momentum) * hard_feat_event.to(ctx.features_event.dtype)
            ctx.features_event[index + half_dim_event] = F.normalize(ctx.features_event[index + half_dim_event], dim=0)

        return grad_inputs_rgb, grad_inputs_event, None, None, None, None, None


def cm_mix_interactive(inputs_rgb, inputs_event, indexes, features_rgb, features_event, momentum=0.5, alpha=0.1):
    return CM_Mix_Interactive.apply(
        inputs_rgb,
        inputs_event,
        indexes,
        features_rgb,
        features_event,
        torch.tensor([momentum], device=inputs_rgb.device, dtype=inputs_rgb.dtype),
        torch.tensor([alpha], device=inputs_rgb.device, dtype=inputs_rgb.dtype),
    )


class CM_Mix_mean_hard(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, indexes, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, indexes)
        return inputs.mm(ctx.features.to(inputs.dtype).t())

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, indexes = ctx.saved_tensors
        half_dim = len(ctx.features) // 2

        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features.to(grad_outputs.dtype))

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, indexes.tolist()):
            batch_centers[index].append(instance_feature)

        for index, feat_list in batch_centers.items():
            feats = torch.stack(feat_list, dim=0)
            mean_feat = feats.mean(dim=0).to(ctx.features.dtype)
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * mean_feat
            ctx.features[index] /= (ctx.features[index].norm() + 1e-8)

            target_center = ctx.features[index].to(feats.dtype).unsqueeze(0)
            sims = torch.mm(feats, target_center.t()).squeeze(1)
            hard_feat = feats[torch.argmin(sims)].to(ctx.features.dtype)
            ctx.features[index + half_dim] = ctx.features[index + half_dim] * ctx.momentum + (1 - ctx.momentum) * hard_feat
            ctx.features[index + half_dim] /= (ctx.features[index + half_dim].norm() + 1e-8)

        return grad_inputs, None, None, None


def cm_mix(inputs, indexes, features, momentum=0.5):
    return CM_Mix_mean_hard.apply(
        inputs,
        indexes,
        features,
        torch.tensor([momentum], device=inputs.device, dtype=inputs.dtype),
    )


class MABSingleMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2):
        super().__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.momentum = momentum
        self.temp = temp
        self.register_buffer("features", torch.randn(num_samples * 2, num_features))
        self.features = F.normalize(self.features, dim=1)

    def forward(self, inputs, targets):
        inputs = F.normalize(inputs, dim=1)
        outputs = cm_mix(inputs, targets, self.features, self.momentum)
        outputs /= self.temp
        mean_logits, hard_logits = torch.chunk(outputs, 2, dim=1)
        loss_mean = F.cross_entropy(mean_logits, targets)
        loss_hard = F.cross_entropy(hard_logits, targets)
        return 0.5 * (loss_mean + loss_hard)


class MABDualMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, alpha=0.1):
        super().__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.momentum = momentum
        self.temp = temp
        self.alpha = alpha
        self.register_buffer("features_rgb", torch.randn(num_samples * 2, num_features, dtype=torch.float32))
        self.register_buffer("features_event", torch.randn(num_samples * 2, num_features, dtype=torch.float32))
        self.features_rgb = F.normalize(self.features_rgb, dim=1)
        self.features_event = F.normalize(self.features_event, dim=1)

    def forward(self, inputs_rgb, inputs_event, targets):
        inputs_rgb = F.normalize(inputs_rgb, dim=1)
        inputs_event = F.normalize(inputs_event, dim=1)

        outputs_rgb, outputs_event = cm_mix_interactive(
            inputs_rgb,
            inputs_event,
            targets,
            self.features_rgb,
            self.features_event,
            self.momentum,
            self.alpha,
        )

        outputs_rgb /= self.temp
        outputs_event /= self.temp

        mean_logits_rgb, hard_logits_rgb = torch.chunk(outputs_rgb, 2, dim=1)
        mean_logits_event, hard_logits_event = torch.chunk(outputs_event, 2, dim=1)

        loss_rgb = 0.5 * (
            F.cross_entropy(mean_logits_rgb, targets) + F.cross_entropy(hard_logits_rgb, targets)
        )
        loss_event = 0.5 * (
            F.cross_entropy(mean_logits_event, targets) + F.cross_entropy(hard_logits_event, targets)
        )
        return (loss_rgb + loss_event) / 2.0
