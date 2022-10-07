import torch
import math


class CombinedMarginLoss(torch.nn.Module):
    def __init__(self, 
                 s, 
                 m1,
                 m2,
                 m3,
                 interclass_filtering_threshold=0):
        super().__init__()
        self.s = s   # hypersphere radius
        self.m1 = m1
        self.m2 = m2 # 指定輸入margin 0.5
        self.m3 = m3
        self.interclass_filtering_threshold = interclass_filtering_threshold
        
        # For ArcFace
        self.cos_m = math.cos(self.m2)
        self.sin_m = math.sin(self.m2)
        self.theta = math.cos(math.pi - self.m2)
        self.sinmm = math.sin(math.pi - self.m2) * self.m2
        self.easy_margin = False


    def forward(self, logits, labels):
        index_positive = torch.where(labels != -1)[0] # 標記可訓練的正樣本索引 0~7

        if self.interclass_filtering_threshold > 0:
            with torch.no_grad():
                dirty = logits > self.interclass_filtering_threshold
                dirty = dirty.float()
                mask = torch.ones([index_positive.size(0), logits.size(1)], device=logits.device)
                mask.scatter_(1, labels[index_positive], 0)
                dirty[index_positive] *= mask
                tensor_mul = 1 - dirty    
            logits = tensor_mul * logits

        target_logit = logits[index_positive, labels[index_positive].view(-1)] # target_logit: 抽取該類別對應的prob score, cos(targetθ)

        if self.m1 == 1.0 and self.m3 == 0.0: # 在角度上加margin (ArcFace)           
            sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
            # cos(targetθ + margin) = cos_t * cos(margin_m) - sin_t * sin(margin_m) 
            cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  
            if self.easy_margin:
                final_target_logit = torch.where(
                    target_logit > 0, cos_theta_m, target_logit)
            else:
                # 根据条件，返回从x,y中选择元素所组成的张量。如果满足条件，则返回x中元素。若不满足，返回y中元素
                # If cos(targetθ) > cos(PI - margin) => T: cos(targetθ + margin_m), F: cos(targetθ) - margin_m(sin(margin_m))
                final_target_logit = torch.where(
                    target_logit > self.theta, cos_theta_m, target_logit - self.sinmm)
            logits[index_positive, labels[index_positive].view(-1)] = final_target_logit # 更新該類別對應的prob score 為 final_target_logit
            logits = logits * self.s                                        # ArcFace Decision Boundaries: s(cos(θ1 + m) − cosθ2) 
        
        elif self.m3 > 0: # 在target_logit上加margin
            final_target_logit = target_logit - self.m3                     # 該類別對應的prob score強制減去margin
            logits[index_positive, labels[index_positive].view(-1)] = final_target_logit # 更新該類別對應的prob score 為 final_target_logit
            logits = logits * self.s                                        # Decision Boundaries: s(cos(targetθ) - m) 
        else:
            raise        

        return logits

class ArcFace(torch.nn.Module): # Angular Magrin
    """ ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """
    def __init__(self, s=64.0, margin=0.5):
        super(ArcFace, self).__init__()
        self.scale = s
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.theta = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin
        self.easy_margin = False


    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
        if self.easy_margin:
            final_target_logit = torch.where(
                target_logit > 0, cos_theta_m, target_logit)
        else:
            final_target_logit = torch.where(
                target_logit > self.theta, cos_theta_m, target_logit - self.sinmm)

        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * self.scale
        return logits

class CosFace(torch.nn.Module): # Additive Magrin
    def __init__(self, s=64.0, m=0.40):
        super(CosFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]
        final_target_logit = target_logit - self.m      # cos(target-margin)
        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * self.s
        return logits

