# Strange attractors
## Finding beauty within chaos!

This repo contains a python script for automatically detecting "stable" chaotic systems.
The script can search for them, store their initial configuration and later displaying the
system after N iterations in an artistic style.

### Installing

```
git clone git@github.com:AIRLegend/attractors.git
cd attractors
pip install -r requirements.txt
```

### Searching

```
python attractors.py search --n_searches 1500 --search_out_path ./sols/
```
This command will spawn the search of 1500 chaotic systems and save the initial configurations
next to a simple plot of the result in the `sols/` directory.


### Visualization

After having generated "solution files" you can use them for creating their
visualizations.

```
python search.py display --solution_file ./sols/sol_0_1715774254.128875.json --cmap Blues --figsize 30 30 --alpha 0.7 --n_samples=200000
```

You can also generate an animation with the `video` parameter:

```
python attractors.py video --solution_file ./sols/sol_10_1715885863.353617.json --cmap cividis --figsize 20 20
```

(See help for the extra supported styling options)


### Examples!

You can generate visualizations such as:


![](https://github.com/AIRLegend/attractors/blob/master/examples/example1.png?raw=true)

![](https://github.com/AIRLegend/attractors/blob/master/examples/example2.png?raw=true)

![](https://github.com/AIRLegend/attractors/blob/master/examples/example3.png?raw=true)

![](https://github.com/AIRLegend/attractors/blob/master/examples/example4.png?raw=true)

![](https://github.com/AIRLegend/attractors/blob/master/examples/example5.png?raw=true)

![](https://github.com/AIRLegend/attractors/blob/master/examples/example6.png?raw=true)

### Resources

I've found these resources very interesting. Worth checking out if you want to learn more
about this stuff!

- https://paulbourke.net/fractals/lyapunov/
- https://en.wikipedia.org/wiki/Lyapunov_exponent
- [Automatic Generation of Strange Attractors, Sprott, J. 1993](https://sprott.physics.wisc.edu/pubs/paper203.pdf)
-  https://en.wikipedia.org/wiki/Attractor
