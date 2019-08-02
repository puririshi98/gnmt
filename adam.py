import math
import torch
from .optimizer import Optimizer
import stochround

class Adam(Optimizer):
    r"""Implements Adam algorithm.
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        param_count=0
        for group in self.param_groups:
            
            for p in group['params']:
                if p.grad is None:
                    continue
                param_count+=1
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data,dtype=torch.float)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data,dtype=torch.float)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data,dtype=torch.float)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1



                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)
                # p.data.addcmul_(-step_size,exp_avg,1/denom)
                # exp_avg.mul_(beta1).add_(1 - beta1, grad)
                # exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                # if amsgrad:
                #     # Maintains the maximum of all 2nd moment running avg. till now
                #     torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                #     # Use the max. for normalizing running avg. of gradient


                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']
                # bias_corrected_avg=exp_avg/bias_correction1
                # bias_corrected_avg_sq=exp_avg_sq/bias_correction2
                # p.data.addcdiv_(-group['lr'],bias_corrected_avg,bias_corrected_avg_sq.sqrt().add_(group['eps']))

                #p.data.addcdiv_(-step_size, exp_avg, denom)
                
                # if state['step']>0 and param_count==11:
                #     print("step==="+str(state['step']))
                #     print("param# :")
                #     print(param_count)
                #     # print('eps:')
                #     # print(group['eps'])
                #     print('step_size:')
                #     print(step_size)
                #     print("1/denom:")
                #     print(torch.norm(1/denom))
                #     # print('denom:')

                #     # print(torch.norm(denom))
                #     # print('exp_avg:')
                #     # print(torch.norm(exp_avg))
                #     print("\n \n \n _________")

                # if state['step']>5:
                #     print(torch.norm(exp_avg))
                # if ((1/denom)>65536).any():
                #     print(state['step'])
                #     print((1/denom)[(1/denom)>65536])
                #     raise ValueError("1/denom exploded")
                # if (exp_avg_sq<0.0).any():
                #     raise ValueError("exp_avg_sq has a negative value after rounding")
                # if torch.isnan(exp_avg_sq).any():
                #     raise ValueError('exp_avg_sq became nan before rounding on step: '+str(state['step']))
                # if torch.isnan(exp_avg).any():
                #     raise ValueError('exp_avg became nan before rounding on step: '+str(state['step']))
                     
                # if exp_avg.type()!='torch.cuda.FloatTensor':
                #     raise ValueError('exp_avg is not float')
                # if exp_avg_sq.type()!='torch.cuda.FloatTensor':
                #     raise ValueError('exp_avg_sq is not float')
                # if (exp_avg>65536).any():
                #     print(exp_avg[exp_avg>65536])
                #     raise ValueError("vals in exp_avg are too big")
                # if (exp_avg_sq>65536).any():
                #     print(exp_avg_sq[exp_avg_sq>65536])
                #     raise ValueError("vals in exp_avg_sq are too big")
                
            for param in group['params']:
                stochround.stochastic_tensor_round(param, param)
                param_state=self.state[param]
                if len(param_state) !=0:
                    exp_avg, exp_avg_sq = param_state['exp_avg'], param_state['exp_avg_sq']
                    if group['amsgrad']:
                        max_exp_avg_sq = param_state['max_exp_avg_sq']
                        stochround.stochastic_tensor_round(max_exp_avg_sq,max_exp_avg_sq)
                    
                    exp_avg=exp_avg/10000
                    exp_avg_sq=exp_avg_sq/10000
                    stochround.stochastic_tensor_round(exp_avg,exp_avg)
                    stochround.stochastic_tensor_round(exp_avg_sq,exp_avg_sq)
                    exp_avg=exp_avg*10000
                    exp_avg_sq=exp_avg_sq*10000
            # for param in group['params']:
            #     state=self.state[param]
            #     exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            #     if (exp_avg_sq<0).any():
            #         raise ValueError("exp_avg_sq has a negative value after rounding")
            #     if torch.isnan(exp_avg_sq).any():
            #         raise ValueError('exp_avg_sq became nan after rounding on step: '+str(state['step']))
            #     if torch.isnan(exp_avg).any():
            #         raise ValueError('exp_avg became nan after rounding on step: '+str(state['step']))
             
        return loss
