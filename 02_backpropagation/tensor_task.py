# %%
import torch
import numpy as np

# %%
w = torch.tensor(1)
x = torch.tensor(2.0)
t = torch.tensor(np.float32(3))
b = torch.tensor(4, dtype=torch.float32)

print('- dtypes')
print(f'w: {w.dtype}')
print(f'x: {x.dtype}')
print(f't: {t.dtype}')
print(f'b: {b.dtype}')

# %%
# modify so all dtypes are torch.float32
w = torch.tensor(1, dtype=torch.float32)
x = torch.tensor(2.0)
t = torch.tensor(np.float32(3))
b = torch.tensor(4, dtype=torch.float32)

print('- modified dtypes')
print(f'w: {w.dtype}')
print(f'x: {x.dtype}')
print(f't: {t.dtype}')
print(f'b: {b.dtype}')

# %%
# (1)
w.requires_grad = True

# %%
a = x + b
y = max(a * w, 0)
l = (y - t).pow(2) + w.pow(2)

print('- grad_fn')
print(f'a: {a.grad_fn}')
print(f'y: {y.grad_fn}')
print(f'l: {l.grad_fn}')

# %%
# we have only one w.r.t. input so take the first element of the tuple
# retain_graph=True is needed so that we can do backward again without a new forward pass
dl_dy = torch.autograd.grad(l, inputs=[y], retain_graph=True)[0]
print(f'gradient of l w.r.t. y: {dl_dy}')

# %%
# why we do not need to set inner nodes to None as well?
w.grad = None
l.backward()
print(f'gradient of l w.r.t. w {w.grad}')

# %%
w.data -= 0.1 * w.grad
w.grad = None
print(f'new w: {w.data}')

# %%
w = torch.tensor(1.0, requires_grad=True)


def loss(w):
    x = torch.tensor(2.0)
    b = torch.tensor(3.0)
    a = x + b
    y = torch.exp(w)
    l = (y - a)**2
    # y/=2
    del y, x, b, w
    return l


loss(w).backward()
print(w.grad)
