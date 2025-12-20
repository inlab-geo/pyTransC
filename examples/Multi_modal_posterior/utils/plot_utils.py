"""Utility functions for plotting."""

import arviz as az
import corner
import matplotlib.pyplot as plt
import numpy as np

from .IO_utils import create_InferenceData_object_per_state


def plot_sample(smpl, zmax):
    """TODO: add docstring."""
    n, msft, d, c = smpl
    depth = [d[0]]
    resistivity = [1 / 10 ** c[0]]
    for i in range(1, n):
        depth.append(d[i])
        depth.append(d[i])
        resistivity.append(1 / 10 ** c[i - 1])
        resistivity.append(1 / 10 ** c[i])
    depth.append(zmax * 2)
    resistivity.append(1 / 10 ** c[-1])

    plt.figure(figsize=(2, 4))
    plt.gca().invert_yaxis()
    plt.plot(resistivity, depth)
    plt.ylabel("depth (m)")
    plt.xlabel("resistvity (Ω m)")
    plt.ylim([zmax, 0])
    return


def plot_mean(fn, zmax):
    """TODO: add docstring."""
    fh = open(fn)
    lines = fh.readlines()
    fields = lines[0].split()
    depth = np.array(fields[17:137], dtype=float)

    mean = 1.0 / np.array(fields[137:257], dtype=float)

    depth = -depth

    plt.figure(figsize=(2, 4))
    plt.gca().invert_yaxis()
    plt.plot(mean, -depth)
    plt.ylabel("depth (m)")
    plt.xlabel("resistvity (Ω m)")
    plt.ylim([zmax, 0])
    # print('horizontal distance: ',x)


def get_sample(smpl, zmax):
    """TODO: add docstring."""
    n, msft, d, c = smpl
    depth = [d[0]]
    resistivity = [1 / 10 ** c[0]]
    for i in range(1, n):
        depth.append(d[i])
        depth.append(d[i])
        resistivity.append(1 / 10 ** c[i - 1])
        resistivity.append(1 / 10 ** c[i])
    depth.append(zmax * 2)
    resistivity.append(1 / 10 ** c[-1])
    return resistivity, depth


def plot_histogram(
    chain,
    xmin,
    xmax,
    nx,
    zmin,
    zmax,
    nz,
    cmap="viridis",
    log=False,
    cmapmin=0.0,
    cmapmax=1.0,
    returndata=False,
    figsize=(2, 4),
    density_cutoff=None,
    pcentiles=None,
    interfaceplotcolor=None,
    density=None,
    verbose=False,
    filename=None,
):
    """TODO: add docstring."""
    # xedges = np.logspace(np.log10(xmin), np.log10(xmax), nx)
    xedges = np.linspace(xmin, xmax, nx)
    yedges = np.linspace(zmin, zmax, nz)

    if density is not None:
        HH = density
    else:
        if verbose:
            print("Start density calculation")
        for idx, piece in enumerate(chain):
            n, msft, d, c = piece
            dd = []
            val = 1.0 / 10 ** c[0]
            dd.append([val, d[0]])
            for i in range(1, n):
                val = 1.0 / 10 ** c[i - 1]
                dd.append([val, d[i]])
                val = 1.0 / 10 ** c[i]
                dd.append([val, d[i]])
            dd.append([val, zmax])
            dd = np.array(dd)
            xxx = []
            yyy = []

            for i in range(len(yedges) - 1):
                y = (yedges[i] + yedges[i + 1]) / 2.0
                x = np.interp(y, dd[:, 1], dd[:, 0])
                xxx.append(x)
                yyy.append(y)

            for i in range(1, len(dd[:, 0]) - 1, 2):
                x0 = min(dd[i, 0], dd[i + 1, 0])
                x1 = max(dd[i, 0], dd[i + 1, 0])
                for j in range(len(xedges) - 1):
                    x = (xedges[j] + xedges[j + 1]) / 2.0
                    if x >= x0 and x <= x1:
                        xxx.append(x)
                        yyy.append(dd[i, 1])

            H, xedges, yedges = np.histogram2d(xxx, yyy, bins=(xedges, yedges))

            if idx == 0:
                HH = np.clip(H, 0, 1)
            else:
                HH = HH + np.clip(H, 0, 1)

        if verbose:
            print("End density calculation")

    hmax = HH.max()
    if log:
        HHplot = np.zeros_like(HH)
        mask = HH != 0
        HHplot[mask] = np.log(HH[mask] / hmax)  # map colormap to a log of density
        HHplot[~mask] = -np.inf
        HHplot = np.transpose(HHplot)
    else:
        HHplot = np.copy(np.transpose(HH)) / hmax

    if (
        density_cutoff is not None
    ):  # white out all cells with density below n% of maximum
        # (if(log) then density_cutoff = np.log(n/100); othrwise density_cutoff=n/100)
        cutoff = density_cutoff / 100
        if log:
            cutoff = np.log(cutoff)
        HHplot[np.where(HHplot <= cutoff)] = -np.inf

    nz, nx = np.shape(HHplot)
    X, Y = np.meshgrid(np.linspace(xmin, xmax, nx), np.linspace(0, zmax, nz))

    if len(chain[0][2]) == 1:
        interfaceplotcolor = None
    if interfaceplotcolor is not None:
        # fig, (ax,ax2) = plt.subplots(1,2,figsize=figsize,width_ratios=[3.,1.])
        gs_kw = {"width_ratios": [3, 1]}
        fig, (ax, ax2) = plt.subplots(
            ncols=2,
            nrows=1,
            figsize=figsize,
            constrained_layout=True,
            sharey=True,
            gridspec_kw=gs_kw,
        )
    else:
        fig, ax = plt.subplots(figsize=figsize)

    ax.invert_yaxis()
    ax.set_ylabel("depth (m)")
    ax.set_xlabel("resistvity (Ω m)")
    n_levels = 99  # number of contour levels to plot
    ax.contourf(X, Y, HHplot, n_levels, cmap=cmap)
    ax.set_ylim([zmax, 0])
    ax.set_xlim([xmin, xmax])

    # plot percentile models on top
    z = np.linspace(0, zmax, nz)
    if pcentiles is not None:
        s = np.zeros((len(chain), nz))
        for i in range(len(chain)):
            r, d = get_sample(chain[i], 500)
            for j, x in enumerate(z):
                s[i, j] = r[d.index(next(obj for obj in d if obj >= x))]
        p = np.percentile(s, pcentiles[0], axis=0)
        for i, x in enumerate(p):
            ax.plot(x, z, pcentiles[1][i], lw=pcentiles[2][i])

    # plot density of interfaces
    if interfaceplotcolor is not None:
        X, Y = np.meshgrid(np.linspace(xmin, xmax, 1), np.linspace(0, zmax, nz))
        dz = zmax / nz
        den = np.zeros(nz)
        for i in range(len(chain)):
            r, d = get_sample(chain[i], 500)
            for j in range(1, len(d)):
                k = int(d[j] / dz)
                if k < nz:
                    den[k] += 1
        den /= np.max(den)
        ax2.fill_betweenx(z, den, color=interfaceplotcolor)

    if filename is not None:
        plt.savefig(filename)

    return HH


def plot_histogram_is(  # noqa: C901
    transd_ensemble,
    xmin,
    xmax,
    nx,
    zmin,
    zmax,
    nz,
    cmap="viridis",
    log=False,
    cmapmin=0.0,
    cmapmax=1.0,
    returndata=False,
    figsize=(2, 4),
    density_cutoff=None,
    pcentiles=None,
    interfaceplotcolor=None,
    density=None,
    verbose=False,
    filename=None,
):
    """TODO: add docstring."""
    # xedges = np.logspace(np.log10(xmin), np.log10(xmax), nx)
    xedges = np.linspace(xmin, xmax, nx)
    yedges = np.linspace(zmin, zmax, nz)

    if density is not None:
        HH = density
        idx = 0
        for state, _ in enumerate(transd_ensemble):
            idx += len(transd_ensemble[state])
    else:
        if verbose:
            print("Start density calculation")
        # for idx,piece in enumerate(chain):
        idx = 0
        for state, x in enumerate(transd_ensemble):
            n = state
            for i, y in enumerate(x):
                d = [0.0, *y[:n]]
                c = y[n:]

                dd = []
                val = 1.0 / 10 ** c[0]
                dd.append([val, d[0]])
                for i in range(1, n):
                    val = 1.0 / 10 ** c[i - 1]
                    dd.append([val, d[i]])
                    val = 1.0 / 10 ** c[i]
                    dd.append([val, d[i]])
                dd.append([val, zmax])
                dd = np.array(dd)
                xxx = []
                yyy = []

                for i in range(len(yedges) - 1):
                    y = (yedges[i] + yedges[i + 1]) / 2.0
                    x = np.interp(y, dd[:, 1], dd[:, 0])
                    xxx.append(x)
                    yyy.append(y)

                for i in range(1, len(dd[:, 0]) - 1, 2):
                    x0 = min(dd[i, 0], dd[i + 1, 0])
                    x1 = max(dd[i, 0], dd[i + 1, 0])
                    for j in range(len(xedges) - 1):
                        x = (xedges[j] + xedges[j + 1]) / 2.0
                        if x >= x0 and x <= x1:
                            xxx.append(x)
                            yyy.append(dd[i, 1])

                H, xedges, yedges = np.histogram2d(xxx, yyy, bins=(xedges, yedges))

                if idx == 0:
                    HH = np.clip(H, 0, 1)
                else:
                    HH = HH + np.clip(H, 0, 1)
                idx += 1

        if verbose:
            print("End density calculation")

    hmax = HH.max()
    if log:
        HHplot = np.zeros_like(HH)
        mask = HH != 0
        HHplot[mask] = np.log(HH[mask] / hmax)  # map colormap to a log of density
        HHplot[~mask] = -np.inf
        HHplot = np.transpose(HHplot)
    else:
        HHplot = np.copy(np.transpose(HH)) / hmax

    if (
        density_cutoff is not None
    ):  # white out all cells with density below n% of maximum
        # (if(log) then density_cutoff = np.log(n/100); othrwise density_cutoff=n/100)
        cutoff = density_cutoff / 100
        if log:
            cutoff = np.log(cutoff)
        HHplot[np.where(HHplot <= cutoff)] = -np.inf

    nz, nx = np.shape(HHplot)
    X, Y = np.meshgrid(np.linspace(xmin, xmax, nx), np.linspace(0, zmax, nz))

    # if(len(chain[0][2])==1): interfaceplotcolor = None
    if interfaceplotcolor is not None:
        # fig, (ax,ax2) = plt.subplots(1,2,figsize=figsize,width_ratios=[3.,1.])
        gs_kw = {"width_ratios": [3, 1]}
        fig, (ax, ax2) = plt.subplots(
            ncols=2,
            nrows=1,
            figsize=figsize,
            constrained_layout=True,
            sharey=True,
            gridspec_kw=gs_kw,
        )
    else:
        fig, ax = plt.subplots(figsize=figsize)

    ax.invert_yaxis()
    ax.set_ylabel("depth (m)")
    ax.set_xlabel("resistvity (Ω m)")
    n_levels = 99  # number of contour levels to plot
    ax.contourf(X, Y, HHplot, n_levels, cmap=cmap)
    ax.set_ylim([zmax, 0])
    ax.set_xlim([xmin, xmax])

    # plot percentile models on top
    z = np.linspace(0, zmax, nz)
    if pcentiles is not None:
        np.percentile(HH, pcentiles[0], axis=0, method="median_unbiased")
        s = np.zeros((idx, nz))
        ii = 0
        for state, x in enumerate(transd_ensemble):
            n = state
            for y in x:
                d = [0.0, *y[:n]]
                c = y[n:]
                r, dp = get_sample_is(n + 1, d, c, 500)
                for j, xx in enumerate(z):
                    s[ii, j] = r[dp.index(next(obj for obj in dp if obj >= xx))]
                ii += 1
        p = np.percentile(s, pcentiles[0], axis=0)
        for i, x in enumerate(p):
            ax.plot(x, z, pcentiles[1][i], lw=pcentiles[2][i])

    # plot density of interfaces
    if interfaceplotcolor is not None:
        X, Y = np.meshgrid(np.linspace(xmin, xmax, 1), np.linspace(0, zmax, nz))
        dz = zmax / nz
        den = np.zeros(nz)
        for state, x in enumerate(transd_ensemble):
            n = state
            for y in x:
                d = [0.0, *y[:n]]
                c = y[n:]
                r, dp = get_sample_is(n + 1, d, c, 500)
                for j in range(1, len(dp)):
                    k = int(dp[j] / dz)
                    if k < nz:
                        den[k] += 1
        den /= np.max(den)
        ax2.fill_betweenx(z, den, color=interfaceplotcolor)

    if filename is not None:
        plt.savefig(filename)

    return HH


def get_sample_is(n, d, c, zmax):
    """TODO: add docstring."""
    # n,msft,d,c=smpl
    depth = [d[0]]
    resistivity = [1 / 10 ** c[0]]
    for i in range(1, n):
        depth.append(d[i])
        depth.append(d[i])
        resistivity.append(1 / 10 ** c[i - 1])
        resistivity.append(1 / 10 ** c[i])
    depth.append(zmax * 2)
    resistivity.append(1 / 10 ** c[-1])
    return resistivity, depth


def cornerplot(
    arviz_data,
    title=None,
    truths=None,
    params=None,
    plotall=False,
    returnnames=False,
    figsize=None,
    titlefontsize=18,
    label_kwargs=None,
):
    """TODO: add docstring."""
    contour_kwargs = {"linewidths": 0.5}
    data_kwargs = {"color": "darkblue"}
    data_kwargs = {"color": "slateblue"}
    # label_kwargs = {"fontsize" : fontsize}
    allvars = list(arviz_data.posterior.keys())
    if params is None:
        var_names = allvars
        if truths is not None:
            truths = {
                var_names[i]: truths[i] for i in range(len(truths))
            }  # construct dictionary of truth values for plotting
    else:
        var_names = [
            allvars[i] for i in params
        ]  # construct list of variable names to plot
        if truths is not None:
            truths = {
                var_names[i]: truths[params[i]] for i in range(len(params))
            }  # construct dictionary of truth values for plotting

    if plotall:
        pass
    #elif truths is not None:
    #    truths = {
    #        var_names[i]: truths[params[i]] for i in range(len(params))
    #   }  # construct dictionary of truth values for plotting

    if figsize is not None:
        fig = plt.figure(figsize=(10, 10))
        corner.corner(
            arviz_data,
            truths=truths,
            fig=fig,
            var_names=var_names,
            bins=40,
            hist_bin_factor=2,
            smooth=True,
            contour_kwargs=contour_kwargs,
            data_kwargs=data_kwargs,
            label_kwargs=label_kwargs,
        )
    else:
        fig = corner.corner(
            arviz_data,
            truths=truths,
            var_names=var_names,
            bins=40,
            hist_bin_factor=2,
            smooth=True,
            contour_kwargs=contour_kwargs,
            data_kwargs=data_kwargs,
            label_kwargs=label_kwargs,
        )
    
    # Adjust layout to make room for the title
    #print('Title cornerplot = ',title)
    if(title is not None):
        fig.tight_layout(rect=[0, 0.03, 1, 0.95]) 
        fig.suptitle(title, fontsize=titlefontsize, y=1.0)
    
    if returnnames:
        return fig, var_names, truths
    return fig


def plot_ensembles(
    azobj,
    params=None,
    truths=None,
    figsize=None,
    traceplot=True,
    filename=None,
    title=None,
    titlefontsize=18,
    label_kwargs=None,
):
    """TODO: add docstring."""
    #print('Title plot_ensembles = ',title)
    fig, v, t = cornerplot(
        azobj,
        truths=truths,
        params=params,
        returnnames=True,
        title=title,
        titlefontsize=titlefontsize,
        figsize=figsize,
        label_kwargs=label_kwargs,
    )

    # traceplot
    if traceplot:
        az.style.use("arviz-doc")
        az.plot_trace(azobj, var_names=v, figsize=fig.get_size_inches())
    if filename is not None:
        plt.savefig(filename)
    plt.show()


def plot_pseudo_samples(
    pseudo_prior_function,
    state,
    ndims,
    ns=10000,
    truths=None,
    params=None,
    figsize=None,
    filename=None,
    title=None,
    titlefontsize=18,
    label_kwargs=None,
):
    """TODO: add docstring."""
    s = np.zeros((1, ns, ndims[state]))
    lp = np.zeros(ns)
    for i in range(ns):
        lp[i], s[0, i, :] = pseudo_prior_function(None, state, returndeviate=True)

    azobj = create_InferenceData_object_per_state(
        s, lp, ndims[state], state
    )  # create list of arviz objetcs to help with plotting
    plot_ensembles(
        azobj,
        params=params,
        truths=truths,
        figsize=figsize,
        traceplot=False,
        filename=filename,
        title=title,
        titlefontsize=titlefontsize,
        label_kwargs=label_kwargs,
    )

def generate_and_plot_samples(
    log_posterior,
    state,
    ndims,
    ns=10000,
    truths=None,
    params=None,
    figsize=None,
    filename=None,
    title=None,
    titlefontsize=18,
    label_kwargs=None,
    returnmaps=False,
):
    """TODO: add docstring."""
    s = np.zeros((1, ns, ndims[state]))
    lp = np.zeros(ns)
    for i in range(ns):
        x = log_posterior.draw_deviate(state)
        s[0, i, :] = x
        lp[i] = log_posterior(x,state)

    azobj = create_InferenceData_object_per_state(
        s, lp, ndims[state], state
    )  # create list of arviz objetcs to help with plotting
    #print('Title generate_and_plot_samples = ',title)
    plot_ensembles(
        azobj,
        params=params,
        truths=truths,
        figsize=figsize,
        traceplot=False,
        title=title,
        filename=filename,
        titlefontsize=titlefontsize,
        label_kwargs=label_kwargs,
    )
    return lp,s
