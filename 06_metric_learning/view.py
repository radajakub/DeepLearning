# %%
""" Resize the notbook to full width, to fit more code and images """
from lab import evaluate_mAP, evaluate_AP
from lab import get_features, distances
from lab import test_set, load_net
import matplotlib.pyplot as plt
from lovely_numpy import lo
import lovely_tensors as lt
import torch
import numpy as np
from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

""" some basic packages and settings to show images inline """

lt.monkey_patch()

%matplotlib inline

""" automatically reload included modules (need to run import command to trigger reloading) """
%load_ext autoreload
%autoreload 2

""" Controls for figure sizes to change """
plt.rcParams.update({'errorbar.capsize': 1})

# %% [markdown]
# ## Retrieval

# %%

# net = load_net('./models/net_class.pl')  # load pretrained classification network
net = load_net('./models/net_triplet.pl')  # load network trained with triplet loss
# net = load_net('./models/net_smoothAP.pl') # load network trained with smoothAP loss

# %% [markdown]
# ### Show nearest neighbours

# %%
np.random.seed(0)
torch.manual_seed(0)

# indices of query images
query_idxs = np.random.choice(len(test_set), size=10, replace=False)

# extract features for all test_samples
loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, num_workers=0)
features, labels = get_features(net, loader, len(test_set))

dists = distances(features, features)

# %%


def show_nearest(test_set, labels, query_idxs, dists):
    # show 50 nearest retrived images for every query in query_idxs
    N = 50

    f, axarr = plt.subplots(len(query_idxs), 1, gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(16, 4), dpi=200)

    num_correct = np.zeros((len(query_idxs), ), dtype=np.int32)

    for i, qidx in enumerate(query_idxs):
        ax = axarr[i]
        d = dists[qidx, :]
        ds = torch.argsort(d)[:N+1]     # N+1 because the query image is retrieved as well
        img = np.ones((28, 28*(N+1)+10, 3))
        sid = 0
        qimg = ((test_set[qidx][0] * 0.5) + 0.5).cpu().detach().numpy()
        img[:, sid:sid+28, :] = np.stack((qimg, qimg, qimg), axis=-1)   # expand to three channels
        qlab = labels[qidx]
        sid += 28+10
        for di in ds:
            if di == qidx:  # skip the query image
                continue
            retrieved_img = ((test_set[di][0] * 0.5) + 0.5).cpu().detach().numpy()[0, :, :]
            retrieved_img = np.stack((retrieved_img, retrieved_img, retrieved_img), axis=-1)
            # start with a black frame
            retrieved_img[:1, :, :] = 0
            retrieved_img[-1:, :, :] = 0
            retrieved_img[:, :1, :] = 0
            retrieved_img[:, -1:, :] = 0
            lab = labels[di]
            if lab == qlab:     # correct -> green frame
                retrieved_img[:1, :, 1] = 1
                retrieved_img[-1:, :, 1] = 1
                retrieved_img[:, :1, 1] = 1
                retrieved_img[:, -1:, 1] = 1
                num_correct[i] += 1
            else:       # incorrect -> red frame
                retrieved_img[:1, :, 0] = 1
                retrieved_img[-1:, :, 0] = 1
                retrieved_img[:, :1, 0] = 1
                retrieved_img[:, -1:, 0] = 1
            img[:, sid:sid+28] = retrieved_img
            sid += 28
        ax.imshow(img)
        ax.axis('off')

    return num_correct


# %%
num_correct = show_nearest(test_set, labels, query_idxs, dists)
print(f'num_correct: {num_correct}')

# %%
mAP, mPrec, mRec = evaluate_mAP(net, test_set)
print(f"{mAP:.2f}")

# %%

# %%
# TODO: plot Precision vs. Recall for all three models

net = load_net('./models/net_class.pl')  # load pretrained classification network
mAP_clf, mPrec_clf, mRec_clf = evaluate_mAP(net, test_set)
net = load_net('./models/net_triplet.pl')  # load network trained with triplet loss
mAP_triplet, mPrec_triplet, mRec_triplet = evaluate_mAP(net, test_set)
net = load_net('./models/net_smoothAP.pl')  # load network trained with smoothAP loss
mAP_ap, mPrec_ap, mRec_ap = evaluate_mAP(net, test_set)

fig = plt.figure(figsize=(4, 4), dpi=200)
plt.plot(mRec_clf, mPrec_clf, label="cross-entropy")
plt.plot(mRec_triplet, mPrec_triplet, label="triplet")
plt.plot(mRec_ap, mPrec_ap, label="smoothAP")
plt.legend()
plt.xlabel("recall")
plt.ylabel("precision")

# %%
# plot training histories


def plot_histories(name: str):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=200)
    train_loss = np.load(name + '_train.npy')
    val_loss = np.load(name + '_val.npy')
    xs = list(range(100))
    ax.plot(xs, train_loss, label='training loss')
    ax.plot(xs, val_loss, label='validation loss')
    ax.legend()


# %%
# plot_histories('./models/net_triplet.pl')
plot_histories('./models/net_smoothAP.pl')
