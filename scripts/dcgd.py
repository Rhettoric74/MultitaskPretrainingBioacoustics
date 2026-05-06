import torch
import numpy as np

class DCGD(torch.optim.Optimizer):
    def __init__(
        self,
        optimizer,
        num_pde,
        type='center',
        classifier_param_indices=None,
        predictor_start_idx=None,
        classifier_start_idx=None,
    ):
        defaults = dict()
        super().__init__(optimizer.param_groups, defaults)
        self.optimizer = optimizer
        self.num_pde = num_pde
        self.epsilon = 1e-8
        self.iter = 0
        self.type = type
        self.conflict_TH = 1e-8
        self.Isconflict = False

        self.classifier_param_indices = classifier_param_indices or []
        self.classifier_param_set = set(self.classifier_param_indices)

        self.predictor_start_idx = predictor_start_idx
        self.classifier_start_idx = classifier_start_idx

    def _flatten_grad(self, grads):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad   

    def _get_all_param_info(self):
        """Get shapes and device info for all parameters"""
        shapes = []
        devices = []
        params_list = []
        
        for group in self.optimizer.param_groups:
            for p in group['params']:
                shapes.append(p.shape)
                devices.append(p.device)
                params_list.append(p)
        
        return shapes, devices, params_list

    def zero_grad(self, set_to_none: bool = True):
        return self.optimizer.zero_grad(set_to_none=set_to_none)
    @torch.no_grad()
    def _group_norms(self, params):
        p2 = torch.zeros((), device=params[0].device)
        g2 = torch.zeros_like(p2)
        none = 0
        for p in params:
            p2 += p.detach().float().pow(2).sum()
            if p.grad is None:
                none += 1
            else:
                g2 += p.grad.detach().float().pow(2).sum()
        return p2.sqrt(), g2.sqrt(), none
    @torch.no_grad()
    def step_center(self, losses):
        self.iter += 1
        
        # Get info for all parameters
        all_shapes, all_devices, all_params = self._get_all_param_info()
        num_params = len(all_shapes)
        
        # Track which parameters have PDE gradients
        pde_has_grad = [False] * num_params
        
        # Compute PDE gradients (first num_pde losses)
        with torch.enable_grad():
            pde_loss = sum(losses[:self.num_pde])
        self.zero_grad()
        pde_loss.backward(retain_graph=True)
        
        pde_grad = []
        for idx, p in enumerate(all_params):
            if p.grad is None:
                pde_grad.append(torch.zeros_like(p).to(p.device))
            else:
                pde_grad.append(p.grad.clone())
                pde_has_grad[idx] = True
        flatten_pde_grad = self._flatten_grad(pde_grad)
        
        # Accumulate BC gradients (classifier losses)
        bc_grad_accum = [torch.zeros_like(p).to(p.device) for p in all_params]
        
        for i in range(self.num_pde, len(losses)):
            with torch.enable_grad():
                self.zero_grad()
                losses[i].backward(retain_graph=True)
            
            for idx, p in enumerate(all_params):
                if p.grad is not None:
                    bc_grad_accum[idx] += p.grad.clone()
        
        flatten_bc_grads = self._flatten_grad(bc_grad_accum)
        
        # Compute norms
        bc_norm = torch.norm(flatten_bc_grads, p=2) + self.epsilon
        pde_norm = torch.norm(flatten_pde_grad, p=2) + self.epsilon
        
        bc_pde_dot = flatten_bc_grads.dot(flatten_pde_grad).item()
        cos_val = bc_pde_dot/(bc_norm * pde_norm)
        
        if cos_val < (-1 + self.conflict_TH):
            self.Isconflict = True
        
        center_norm_sq = (2 * (1 + bc_pde_dot/(bc_norm * pde_norm)))
        #print("Center norm:", center_norm_sq)
        center_norm_sq = max(center_norm_sq, 1e-4)
        center_total_dot = flatten_bc_grads.dot(flatten_bc_grads + flatten_pde_grad).item()
        
        # Unflatten gradients
        unflatten_pde_grad = self._unflatten_grad(flatten_pde_grad, all_shapes)
        unflatten_bc_grads = self._unflatten_grad(flatten_bc_grads, all_shapes)
        
        # Apply updates
        with torch.enable_grad():
            self.zero_grad()
            for idx, p in enumerate(all_params):
                if idx in self.classifier_param_set:
                    # Force a clone to ensure gradient is set
                    p.grad = (unflatten_bc_grads[idx] / bc_norm).clone()
                elif pde_has_grad[idx]:
                    p.grad = ((center_total_dot/center_norm_sq) * (
                        (unflatten_bc_grads[idx] / bc_norm) + (unflatten_pde_grad[idx] / pde_norm)
                    )).clone()
                else:
                    p.grad = (unflatten_bc_grads[idx] / bc_norm).clone()
        if self.iter % 1000 == 0 and self.predictor_start_idx is not None:
            all_params = [p for group in self.optimizer.param_groups for p in group["params"]]
            predictor_params = all_params[self.predictor_start_idx:self.classifier_start_idx]
            classifier_params = all_params[self.classifier_start_idx:]
        
            pred_pn, pred_gn, pred_none = self._group_norms(predictor_params)
            cls_pn, cls_gn, cls_none = self._group_norms(classifier_params)
        
            print(
                f"[DCGD] predictor: param_norm={pred_pn.item():.3e}, "
                f"grad_norm={pred_gn.item():.3e}, grad_none={pred_none}"
            )
            print(
                f"[DCGD] classifier: param_norm={cls_pn.item():.3e}, "
                f"grad_norm={cls_gn.item():.3e}, grad_none={cls_none}"
            )

        # clipping on predictor parameters to ensure their stability
        predictor_params = all_params[self.predictor_start_idx:self.classifier_start_idx]
        torch.nn.utils.clip_grad_norm_(predictor_params, max_norm=10.0)
        
        self.optimizer.step()
        return self.Isconflict

    @torch.no_grad()
    def step_avg(self, losses):
        self.iter += 1
        
        # Get info for all parameters
        all_shapes, all_devices, all_params = self._get_all_param_info()
        num_params = len(all_shapes)
        
        # Track which parameters have PDE gradients
        pde_has_grad = [False] * num_params
        
        # Compute PDE gradients
        with torch.enable_grad():
            pde_loss = sum(losses[:self.num_pde])
        self.zero_grad()
        pde_loss.backward(retain_graph=True)
        
        pde_grad = []
        for idx, p in enumerate(all_params):
            if p.grad is None:
                pde_grad.append(torch.zeros_like(p).to(p.device))
            else:
                pde_grad.append(p.grad.clone())
                pde_has_grad[idx] = True
        flatten_pde_grad = self._flatten_grad(pde_grad)
        
        # Accumulate BC gradients
        bc_grad_accum = [torch.zeros_like(p).to(p.device) for p in all_params]
        
        for i in range(self.num_pde, len(losses)):
            with torch.enable_grad():
                self.zero_grad()
                losses[i].backward(retain_graph=True)
            
            for idx, p in enumerate(all_params):
                if p.grad is not None:
                    bc_grad_accum[idx] += p.grad.clone()
        
        flatten_bc_grads = self._flatten_grad(bc_grad_accum)
        
        # Compute combined gradient (simple average)
        flatten_total_grad = (flatten_bc_grads + flatten_pde_grad) / 2
        
        # Compute norms and dot products
        bc_norm = torch.norm(flatten_bc_grads, p=2) + self.epsilon
        pde_norm = torch.norm(flatten_pde_grad, p=2) + self.epsilon
        
        total_bc_dot = flatten_total_grad.dot(flatten_bc_grads).item()
        total_pde_dot = flatten_total_grad.dot(flatten_pde_grad).item()
        bc_pde_dot = flatten_bc_grads.dot(flatten_pde_grad).item()
        
        cos_val = bc_pde_dot/(bc_norm * pde_norm)
        
        if cos_val < (-1 + self.conflict_TH):
            self.Isconflict = True
        
        # Unflatten gradients
        unflatten_pde_grad = self._unflatten_grad(flatten_pde_grad, all_shapes)
        unflatten_bc_grads = self._unflatten_grad(flatten_bc_grads, all_shapes)
        
        # Apply updates
        with torch.enable_grad():
            self.zero_grad()
            DualOut = (total_bc_dot < 0) or (total_pde_dot < 0)
            
            if DualOut:
                for idx, p in enumerate(all_params):
                    if idx in self.classifier_param_set:
                        # Classifier-only parameters just use BC gradient
                        p.grad = unflatten_bc_grads[idx] / bc_norm
                    elif pde_has_grad[idx]:
                        p.grad = ((1 - bc_pde_dot/(bc_norm**2)) * unflatten_bc_grads[idx] + 
                                  (1 - bc_pde_dot/(pde_norm**2)) * unflatten_pde_grad[idx]) / 2
                    else:
                        p.grad = unflatten_bc_grads[idx] / bc_norm
            else:
                # No gradient conflict, use total loss
                total_loss = sum(losses)
                total_loss.backward()
        
        self.optimizer.step()
        return self.Isconflict
    @torch.no_grad()
    def step_proj(self, losses):
        self.iter += 1
        
        # Get info for all parameters
        all_shapes, all_devices, all_params = self._get_all_param_info()
        num_params = len(all_shapes)
        
        # Track which parameters have PDE gradients
        pde_has_grad = [False] * num_params
        
        # Compute PDE gradients
        with torch.enable_grad():
            pde_loss = sum(losses[:self.num_pde])
        self.zero_grad()
        pde_loss.backward(retain_graph=True)
        
        pde_grad = []
        for idx, p in enumerate(all_params):
            if p.grad is None:
                pde_grad.append(torch.zeros_like(p).to(p.device))
            else:
                pde_grad.append(p.grad.clone())
                pde_has_grad[idx] = True
        flatten_pde_grad = self._flatten_grad(pde_grad)
        
        # Accumulate BC gradients
        bc_grad_accum = [torch.zeros_like(p).to(p.device) for p in all_params]
        
        for i in range(self.num_pde, len(losses)):
            with torch.enable_grad():
                self.zero_grad()
                losses[i].backward(retain_graph=True)
            
            for idx, p in enumerate(all_params):
                if p.grad is not None:
                    bc_grad_accum[idx] += p.grad.clone()
        
        flatten_bc_grads = self._flatten_grad(bc_grad_accum)
        flatten_total_grad = (flatten_bc_grads + flatten_pde_grad) / 2
        
        # Compute norms and dot products
        bc_norm = torch.norm(flatten_bc_grads, p=2) + self.epsilon
        pde_norm = torch.norm(flatten_pde_grad, p=2) + self.epsilon
        
        total_bc_dot = flatten_total_grad.dot(flatten_bc_grads).item()
        total_pde_dot = flatten_total_grad.dot(flatten_pde_grad).item()
        bc_pde_dot = flatten_bc_grads.dot(flatten_pde_grad).item()
        
        cos_val = bc_pde_dot/(bc_norm * pde_norm)
        
        if cos_val < (-1 + self.conflict_TH):
            self.Isconflict = True
        
        # Unflatten gradients
        unflatten_pde_grad = self._unflatten_grad(flatten_pde_grad, all_shapes)
        unflatten_bc_grads = self._unflatten_grad(flatten_bc_grads, all_shapes)
        
        # Apply updates
        with torch.enable_grad():
            self.zero_grad()
            if total_bc_dot < 0:
                for idx, p in enumerate(all_params):
                    if idx in self.classifier_param_set:
                        p.grad = unflatten_bc_grads[idx] / bc_norm
                    elif pde_has_grad[idx]:
                        p.grad = (-bc_pde_dot/(bc_norm**2)) * unflatten_bc_grads[idx] + unflatten_pde_grad[idx]
                    else:
                        p.grad = unflatten_bc_grads[idx] / bc_norm
            elif total_pde_dot < 0:
                for idx, p in enumerate(all_params):
                    if idx in self.classifier_param_set:
                        p.grad = unflatten_bc_grads[idx] / bc_norm
                    elif pde_has_grad[idx]:
                        p.grad = unflatten_bc_grads[idx] + (-bc_pde_dot/(pde_norm**2)) * unflatten_pde_grad[idx]
                    else:
                        p.grad = unflatten_bc_grads[idx] / bc_norm
            else:
                total_loss = sum(losses)
                total_loss.backward()
        
        self.optimizer.step()
        return self.Isconflict

    @torch.no_grad()
    def step(self, losses):
        if self.type == 'center':
            return self.step_center(losses)
        elif self.type == 'avg':
            return self.step_avg(losses)
        elif self.type == 'proj':
            return self.step_proj(losses)
        else:
            raise NotImplementedError(f'Unknown type: {self.type}')