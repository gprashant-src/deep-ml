import torch
import triton
import triton.language as tl


@triton.jit
def confidence_prefix_kernel(conf_ptr, cum_ptr, len_ptr, D, threshold,
                             stride_cr, stride_cd, stride_ur, stride_ud):
    # TODO: one program per row; running product, store cum[r, j], count positions >= threshold
    row_idx = tl.program_id(0)

    conf_base_ptr = conf_ptr + row_idx * stride_cr
    cum_base_ptr = cum_ptr + row_idx * stride_ur

    P = 1.0
    count = 0

    for j in range(D):
        c = tl.load(conf_base_ptr + j * stride_cd)

        P *= c

        tl.store(cum_base_ptr + j * stride_ud, P)

        if P >= threshold: count += 1

    tl.store(len_ptr + row_idx, count)



def confidence_prefix(conf: torch.Tensor, threshold: float):
    # TODO: return (cum, lengths)
    R, D = conf.shape

    cum = torch.empty_like(conf)
    lengths = torch.empty((R, ), dtype=torch.int32, device=conf.device)

    grid = (R, )

    confidence_prefix_kernel[grid](conf, cum , lengths, D, threshold, conf.stride(0), conf.stride(1), cum.stride(0), cum.stride(1))

    return cum, lengths


def plan_with_budget(conf: torch.Tensor, budget: int):
    # TODO: threshold = budget-th largest cumulative product; return (lengths, threshold)
    cum = torch.cumprod(conf, dim=-1)

    arr = cum.view(-1)

    k = min(max(budget, 1), arr.numel())

    vals, _ = torch.sort(arr, descending=True)
    threshold = vals[k - 1].item()

    _, lengths = confidence_prefix(conf, threshold)
    return lengths, threshold
