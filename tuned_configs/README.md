## Naming rules
[algo]\_[human_num]\_[method]\_[randomization]\_[scenario]
- algo: happo only
- human\_num: 
    - 5p: 5 humans
    - 10p: 10 humans
- methods: 
    - [number]c: selfplay using discriminator with [number] preference class
    - sp: selfplay without using discriminator
- randomization:
    - rvs: randomize human maximum velocity and size
    - nvs: no randomization on human maximum velocity and size
- scenario:
    - circlecross (cc)
    - room361 (sr)

## Experiment cases
- Evaluating simulation performance w/wo lagrange multiplier
    - create base model for constraint value estimation
        - 5p, nvs, sr: base case
        - 5p, rvs, sr: adapt to randomize human attribute
        - 5p, rvs, cc: adapt to different scenarios
        - 10p, rvs, cc: adapt to scenario with larger human number
    - baseline models:
    - target models:
- Setup your own experiment
    - important parameters:
        - share\_param: use for selfplay. If you want to use discriminator please turn it off. Though training with discriminator also using parameters sharing, we want distinguish these two mode.
        - use\_discriminator: use for our proposed variational exploration but selfplay dont use it, turn it off.
        - human\_preference\_vector\_dim: number of human behavior classes. Defined in env args and will be passed to algo args at running time.
        - human\_random\_pref\_v\_and\_size: randomize human velocity and speed.
        - human\_fov: human field of view (default 110), you can try your own one (based on real world observation). I want try changing it later.