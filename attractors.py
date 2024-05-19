import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import json
import logging
import argparse
import sys

from tqdm import trange, tqdm


logger = logging.getLogger('chaos')

if not logger.hasHandlers():
    stdout_handler = logging.StreamHandler(sys.stdout)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("{asctime} {name} [{levelname}] {message}", style="{")
    stdout_handler.setFormatter(formatter)

    logger.addHandler(stdout_handler)


def plot_system(xs, ys, color, figsize=(8,8), minimal=False, title=None,
                cmap='Blues', bg_color='#0b0b0b', alpha=.3, size=.8):

    cmap = plt.get_cmap(cmap)

    fig, ax = plt.subplots(1, figsize=figsize)

    if isinstance(color, str):
        ax.scatter(xs, ys, s=size, alpha=alpha, marker='.', color='red')
    else:
        ax.scatter(xs, ys, s=size, alpha=alpha, marker='.', c=color, cmap=cmap)

    if not minimal:
        ax.set_title(title)
    else:
        # Disable everything for artistic plot
        fig.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        ax.set_axis_off()


    return fig


def render_video(sol, figsize=(8,8), output='./animation.mp4',
                cmap='Blues', bg_color='#0b0b0b', alpha=.3, size=.8):

    # Perturb N times the initial solution and run the simulation N times
    DELTA = 0.05
    n_iters = 100

    sol['as'] = np.array(sol['as'])
    sol['bs'] = np.array(sol['bs'])

    mods_as = np.linspace(
        np.array(sol['as']),
        np.array(sol['as']) + np.random.uniform(-DELTA, DELTA, sol['as'].shape),
        n_iters
    )

    mods_bs = np.linspace(
        np.array(sol['bs']),
        np.array(sol['bs']) + np.random.uniform(-DELTA, DELTA, sol['bs'].shape),
        n_iters
    )

    alter = np.linspace(-1, 1, n_iters)

    states = []

    logger.info(f"Simulating {n_iters} random variations of the initial configuration.")

    for i, delta in enumerate(tqdm(alter)):

        si = dict(sol)

        si['as'] = mods_as[i]
        si['bs'] = mods_bs[i]

        states.append(
            run_system(si, n_iter=100_000)
        )


    # Remove states that "exploded" so the animation is cleaner
    states = [s for s in states if len(s[0]) > 9_000]

    if len(states) < 100_000 / 2:
        logger.warn("A considerable number of states exploded. The animation will be shorter...")

    # Function to update the plot for each frame
    def update(frame):
        xs = states[frame][0]
        ys = states[frame][1]
        ds = states[frame][2]

        newdata = np.column_stack((xs, ys))

        sc.set_offsets(newdata)

        colors = cmap(ds)
        sc.set_color(colors)

        pbar.update(1)
        return sc,

    cmap = plt.get_cmap('bone')

    fig, ax = plt.subplots(figsize=figsize, facecolor="black")
    fig.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    ax.set_axis_off()

    sc = ax.scatter(np.array(states)[:, 0], np.array(states)[:, 1], alpha=1, cmap=cmap, marker='.', s=.1)

    with tqdm(total=len(states)) as pbar:
        ani = animation.FuncAnimation(fig, update, frames=len(states), interval=90, blit=True)

    logger.info("Saving animation...")
    ani.save(output)


def run_system(solution, n_iter=10_000):
    a = solution['as']
    b = solution['bs']

    distances = []

    x = solution['x0']
    y = solution['y0']

    xs = []
    ys = []

    xs.append(x)
    ys.append(y)
    distances.append(1)


    for i in range(n_iter):
        x = a[0] + a[1] * x  + a[2] * x**2  +a[3]*x*y  +a[4]*y  +a[5]*y**2
        y = b[0] + b[1] * x  + b[2] * x**2  +b[3]*x*y  +b[4]*y  +b[5]*y**2

        d = np.linalg.norm(
            np.array((x,y)) - np.array((xs[-1], ys[-1]))
        )

        if  d > 1e6 or d < 1e-6:
            # Divergence / Convergence
            break

        xs.append(x)
        ys.append(y)
        distances.append(d)

    return xs, ys, distances


def draw_system(solution, n_iter=10_000, title=None, art=False, **kwargs):

    xs, ys, distances = run_system(solution, n_iter=n_iter)

    if not art:
        fig = plot_system(xs, ys, color='red')
    else:
        fig = plot_system(xs, ys, color=distances, **kwargs)

    return fig


def save_solutions(solutions, suffix=None, path='./sols/'):
    plt.ioff()

    logger.info("Saving solutions!")

    if not suffix:
        suffix = time.time()

    for idx, s in enumerate(solutions):
        so = dict()
        so['as'] = s['as'].tolist()
        so['bs'] = s['bs'].tolist()
        so['x0'] = s['x0']
        so['y0'] = s['y0']

        with open(f'{path}sol_{idx}_{suffix}.json', 'w') as f:
            json.dump(so, f)

        fig = draw_system(s, n_iter=5000, art=False)
        fig.savefig(f"{path}sol_{idx}_{suffix}.png")


def search(n_runs=1000, savepath='./sols/'):

    solutions = []

    for _ in trange(n_runs):

        a = np.random.uniform(-1.5, 1.1, 6)
        b = np.random.uniform(-1.5, 1.1, 6)

        distances = []
        lyapunov = 0

        x = 0.00
        y = 0.00

        x_ep = x + np.random.uniform(-1e-5, 1e-5) / 1e-3
        y_ep = y + np.random.uniform(-1e-5, 1e-5) / 1e-3

        # Normal points
        xs = []
        ys = []

        # Alternative points for Lyapunov exponent computation
        xs_ep = []
        ys_ep = []

        xs.append(x)
        ys.append(y)

        xs_ep.append(x_ep)
        ys_ep.append(y_ep)


        distances.append(
            np.linalg.norm(np.array((x,y)) - np.array((x_ep,y_ep)))
        )


        success = True

        for i in range(10_000):
            # Compute curr point
            x = a[0] + a[1] * x  + a[2] * x**2  +a[3]*x*y  +a[4]*y  +a[5]*y**2
            y = b[0] + b[1] * x  + b[2] * x**2  +b[3]*x*y  +b[4]*y  +b[5]*y**2

            # Compute alternative point
            x_ep = a[0] + a[1] * x_ep  + a[2] * x_ep**2  +a[3]*x_ep*y_ep  +a[4]*y_ep  +a[5]*y_ep**2
            y_ep = b[0] + b[1] * x_ep  + b[2] * x_ep**2  +b[3]*x_ep*y_ep  +b[4]*y_ep  +b[5]*y_ep**2


            # Disance between alternative points
            dv_ep = np.array((x_ep, y_ep)) - np.array((x,y))
            d_ep = np.linalg.norm(dv_ep)

            # Disance between consecutive points
            dv = np.array((x, y)) - np.array((xs[-1],ys[-1]))
            d = np.linalg.norm( dv)

            lyapunov += np.log2(abs(d_ep / distances[0]))

            # If divergengence is detected, stop
            if np.isnan(lyapunov) or d < 1e-8 or d > 1e8:
                success = False
                break

            # if lyapunov exponen goes negative, stop
            if (i>=1_000) and (lyapunov/i < 0):
                success = False
                break

            xs.append(x)
            ys.append(y)
            distances.append(d_ep)

            # Move the alt points so hey have the same relative separation as the beginning
            x_ep -= distances[0] * (dv_ep[0] / d_ep)
            y_ep -= distances[0] * (dv_ep[1] / d_ep)


        if success:
            s = {
                'as': np.array(a),
                'bs': np.array(b),
                'x0': xs[0],
                'y0': ys[0],
                'lyapunov': lyapunov
            }

            if s['lyapunov'] < 1e-8:
                success = False
                continue

            logger.info('Found solution!')
            solutions.append(s)


    savepath = savepath + '/' if not savepath.endswith('/') else savepath
    save_solutions(solutions, path=savepath)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Search beauty within chaos.')
    parser.add_argument('mode', default='search', choices=['search', 'display', 'video'],
                        help='Mode of the script. Use `search` first for generating initial solutions and then `display` for generating visualizations.')
    parser.add_argument('--solution_file', default=None,
                        help='Path to json with the initial config of the system. Only used for `display`.')
    parser.add_argument('--cmap', default='blues',
                        help='Colormap name for the visualization. Only used for `display` mode.')
    parser.add_argument('--figsize', nargs=2, type=int ,default=(8,8),
                        help='Size of the figure. Default is 8 8')
    parser.add_argument('--n_samples', default=50_000, type=int,
                        help='Number of iterations for running the system. Only used for `display`.')
    parser.add_argument('--n_searches', default=1000, type=int,
                        help='Number of searches to run. Only used for `search` mode.')
    parser.add_argument('--alpha', default=.3, type=float,
                        help='Transparency of the points. Only used for `display` mode.')
    parser.add_argument('--viz_out_path', default=None, type=str,
                        help='Where to save the visualization')
    parser.add_argument('--search_out_path', default='./sols/', type=str,
                        help='Path where the found initial configurations will be saved. Default is `./sols/.`')


    args = parser.parse_args()

    if args.mode == 'search':
        search(args.n_searches, savepath=args.search_out_path)

    elif args.mode == 'display':
        sol_path = args.solution_file
        if sol_path is None:
            raise ValueError("A path to the solution.json is needed!")

        with open(sol_path, 'r') as f:
            sol = json.load(f)

        xs, ys, distances = run_system(sol, args.n_samples)
        fig = plot_system(xs, ys, color=distances, minimal=True,
                          cmap=args.cmap,
                          figsize=args.figsize,
                          alpha=args.alpha
                          )

        if args.viz_out_path is None:
            suffix = time.time()
            outpath = f"out_{suffix}.png"
        else:
            outpath = args.viz_out_path
        fig.savefig(outpath)

    elif args.mode == 'video':
        sol_path = args.solution_file

        if args.viz_out_path is None:
            suffix = time.time()
            outpath = f"out_{suffix}.mp4"
        else:
            outpath = args.viz_out_path

        if sol_path is None:
            raise ValueError("A path to the solution.json is needed!")

        with open(sol_path, 'r') as f:
            sol = json.load(f)

        render_video(sol,
                     output=outpath,
                     cmap=args.cmap,
                     figsize=args.figsize,
                     )
