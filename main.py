import argparse

import pyro
import pyro.distributions
import pyro.contrib
import torch
import torch.distributions
import torch.distributions.constraints
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import enum
import matplotlib.colors as mcolor
from tqdm import trange


class Kernel(enum.Enum):
    RBF = 1
    MATERN32 = 2
    MATERN52 = 3
    EXP = 4


def kernel(x: torch.Tensor, lengthscale = 1.0, variance = 1.0, epsilon  = 1e-12, f: Kernel = Kernel.RBF):
    scaled_x = x / lengthscale
    x2 = (scaled_x ** 2).sum(-1, keepdim=True)
    xz = scaled_x.matmul(scaled_x.transpose(-2, -1))
    r2 = (x2 - 2 * xz + x2.transpose(-2, -1)).clamp(min=0)
    r = (r2 + epsilon).sqrt()
    if f == Kernel.EXP:
        k = variance * torch.exp(-r)
    elif f == Kernel.MATERN32:
        sqrt3_r = 3 ** 0.5 * r
        k = variance * (1 + sqrt3_r) * torch.exp(-sqrt3_r)
    elif f == Kernel.MATERN52:
        sqrt5_r = 5 ** 0.5 * r
        k = variance * (1 + sqrt5_r + (5 / 3) * r2) * torch.exp(-sqrt5_r)
    elif f == Kernel.RBF:
        k = variance * torch.exp(-0.5 * r2)
    else:
        raise ValueError(f'Invalid kernel type: {f}')
    k = k.contiguous()
    k.diagonal(dim1=-2, dim2=-1)[:] += 1e-2
    return k


def gdrf(*args, **kwargs):
    xs, ws = args
    N = xs.size(0)
    V = ws.size(1)
    K = kwargs['ntopics']
    max_obs = ws.sum(dim=1).max().type(torch.int).detach().cpu().item()

    lengthscale = kwargs['lengthscale']
    variance = kwargs['variance']
    dirichlet_param = kwargs['dirichletparam']
    kernel_type = kwargs['kernel_type']
    dir_p = torch.Tensor([dirichlet_param for _ in range(V)]).cuda()

    covariance = kernel(xs, lengthscale=lengthscale, variance=variance, f=kernel_type)
    lff = torch.linalg.cholesky(covariance)
    mean = torch.zeros(xs.shape[0], dtype=torch.float, device=xs.device).unsqueeze(0)
    with pyro.plate("topics", K, device=xs.device):
        log_topic_prob = pyro.sample("log_topic_prob",
                                     pyro.distributions.MultivariateNormal(mean, scale_tril=lff))
        word_topic_prob = pyro.sample("word_topic_prob", pyro.distributions.Dirichlet(dir_p))
    topic_prob = torch.softmax(log_topic_prob, -2).squeeze()
    word_prob = (topic_prob.transpose(-2, -1) @ word_topic_prob)
    with pyro.plate("words", N, device=xs.device):
        obs = pyro.sample("obs", pyro.distributions.Multinomial(total_count=max_obs, probs=word_prob), obs=ws)
    return obs


def variational_distribution(*args, **kwargs):
    xs, ws = args

    N = xs.size(0)
    V = ws.size(1)
    K = kwargs['ntopics']

    kernel_type = kwargs['kernel_type']
    mean = pyro.param('mean_log_topic_prob', torch.zeros((K, N), dtype=torch.float, device=xs.device))

    lengthscale_constraint = torch.distributions.constraints.nonnegative
    lengthscale_tensor = torch.tensor(kwargs['lengthscale'], dtype=torch.float, device=xs.device).repeat(K)
    lengthscale = pyro.param('lengthscale', lengthscale_tensor, constraint=lengthscale_constraint)

    variance_constraint = torch.distributions.constraints.nonnegative
    variance_tensor = torch.tensor(kwargs['variance'], dtype=torch.float, device=xs.device).repeat(K)
    variance = pyro.param('variance', variance_tensor, constraint=variance_constraint)

    word_topic_prob_constraint = torch.distributions.constraints.stack(
        [torch.distributions.constraints.simplex for _ in range(K)],
        dim=-2
    )
    word_topic_prob_tensor = torch.tensor(kwargs['dirichletparam'], dtype=torch.float, device=xs.device).repeat(K, V)
    word_topic_prob = pyro.param('word_topic_prob_map', word_topic_prob_tensor, constraint=word_topic_prob_constraint)

    with pyro.plate("topics", K, device=xs.device):

        covariance = kernel(
            xs.unsqueeze(-3),
            lengthscale=lengthscale.unsqueeze(-1).unsqueeze(-1),
            variance=variance.unsqueeze(-1).unsqueeze(-1),
            f=kernel_type
        )
        lff = torch.linalg.cholesky(covariance)
        pyro.sample('log_topic_prob', pyro.distributions.MultivariateNormal(mean, scale_tril=lff))
        pyro.sample('word_topic_prob', pyro.distributions.Delta(word_topic_prob).to_event(1))


def topic_probs():
    params = pyro.get_param_store()
    topic_prob = torch.softmax(params['mean_log_topic_prob'], -2).squeeze()
    return topic_prob


def word_topic_probs():
    params = pyro.get_param_store()
    return params['word_topic_prob_map']


def word_probs():
    topic_prob = topic_probs()
    word_topic_prob = word_topic_probs()
    return topic_prob.transpose(-2, -1) @ word_topic_prob


def perplexity(w):
    return ((w * word_probs().log()).sum() / -w.sum()).exp().detach().cpu().item()


def plot_obs(data):
    return stackplot(data, ys=data.div(data.sum(axis=1), axis=0).to_numpy().T)


def plot_words(data):
    return stackplot(data, ys=word_probs().T.detach().cpu().numpy())


def plot_topics(data):
    return stackplot(data, ys=topic_probs().detach().cpu().numpy())


def stackplot(data, ys, dpi=200):
    fig = plt.figure(figsize=(6, 6), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.stackplot(data.index.to_numpy(), ys)
    plt.xticks(rotation=45)
    return fig


def plot_wt_matrix(data):
    return heatmap(data, ys=word_topic_probs().detach().cpu().numpy())


def heatmap(data, ys, dpi=200):
    ntopics = ys.shape[0]
    fig = plt.figure(figsize=(30, 6), dpi=dpi)
    ax = fig.add_subplot(111)
    im = ax.imshow(ys, aspect='auto', interpolation=None, cmap='jet', norm=mcolor.LogNorm(vmin=1e-6, vmax=1))
    fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(data.columns)), labels=data.columns)
    ax.set_yticks(np.arange(ntopics), labels=list(range(1, ntopics + 1)))
    ax.set_xticks(np.arange(len(data.columns) + 1) + .5, minor=True)
    ax.set_yticks(np.arange(ntopics - 1) + .5, minor=True)
    ax.spines[:].set_visible(False)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    plt.xticks(rotation=90)
    fig.tight_layout()
    return fig


def setup_log_dfs(config):
    log_df = dict()
    narr = [None] * config['epochs']
    index = pd.Index(range(config['epochs']), name='epoch')
    t_range = range(1, config['ntopics']+1)
    log_df['metrics'] = pd.DataFrame({'loss': narr, 'perplexity': narr}, dtype=float, index=index)
    log_df['lengthscale'] = pd.DataFrame({f'topic_{k}': narr for k in t_range}, dtype=float, index=index)
    log_df['variance'] = pd.DataFrame({f'topic_{k}': narr for k in t_range}, dtype=float, index=index)
    return log_df


def train(config):
    data = pd.read_csv(config['data'], parse_dates=['sample_time'], index_col='sample_time')
    xs = torch.Tensor(
        [(t.timestamp() - data.index[0].timestamp()) / (data.index[-1].timestamp() - data.index[0].timestamp()) for t in
         data.index]).unsqueeze(1).cuda()
    ws = torch.Tensor(data.to_numpy().astype(int)).cuda()

    model = pyro.poutine.scale(scale=1.0 / len(xs))(gdrf)
    guide = pyro.poutine.scale(scale=1.0 / len(xs))(variational_distribution)

    lr = config['lr']

    pyro.clear_param_store()
    optimizer = pyro.optim.ClippedAdam({"lr": lr})
    objective = pyro.infer.JitTrace_ELBO(num_particles=2, vectorize_particles=True, max_plate_nesting=1)
    svi = pyro.infer.SVI(model, guide, optimizer, loss=objective)

    log_dfs = setup_log_dfs(config)
    pbar = trange(config['epochs'])
    for step in pbar:
        loss = svi.step(xs, ws, ntopics=config['ntopics'], lengthscale=config['lengthscale'], variance=config['variance'],
                        dirichletparam=config['dirichletparam'], kernel_type=config['kernel'])
        pbar.set_description(f'Loss: {loss}')

        params = pyro.get_param_store()

        log_dfs['metrics'].at[step, 'loss'] = loss
        log_dfs['metrics'].at[step, 'perplexity'] = perplexity(ws)
        log_dfs['lengthscale'].loc[step] = params['lengthscale'].detach().cpu().numpy()
        log_dfs['variance'].loc[step] = params['variance'].detach().cpu().numpy()

    os.makedirs(config['out'], exist_ok=True)
    plot_words(data).savefig(os.path.join(config['out'], 'words.png'))
    plot_obs(data).savefig(os.path.join(config['out'], 'obs.png'))
    plot_topics(data).savefig(os.path.join(config['out'], 'topics.png'))
    plot_wt_matrix(data).savefig(os.path.join(config['out'], 'wt_matrix.png'))

    for df_name, df in log_dfs.items():
        df.to_csv(os.path.join(config['out'], df_name + '.csv'))


def main(args):
    train(vars(args))


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help="format: one datetime column labeled 'sample_time', and counts in all others")
    parser.add_argument('--out', help='output folder')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate for SVI')
    parser.add_argument('--epochs', default=1000, type=int, help='Number of training epochs')
    parser.add_argument('--ntopics', default=4, type=int, help='Number of topics to use')
    parser.add_argument('--lengthscale', default=0.001, type=float, help='Initial kernel lengthscale for all topics')
    parser.add_argument('--variance', default=0.1, type=float, help='Initial kernel variance for all topics')
    parser.add_argument('--dirichletparam', default=0.01, type=float, help='Initial dirichlet parameter for all topics')
    parser.add_argument('--kernel', type=Kernel, choices=list(Kernel), default=Kernel.RBF, help='Kernel type')
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    arguments = parser.parse_args()
    main(arguments)
