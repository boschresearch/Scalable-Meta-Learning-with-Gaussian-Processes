# ScaML-GP - Scalable Meta-Learning with Gaussian Processes (AISTATS 2024)

This is the companion code for the benchmarking study reported in the publication
"Scalable Meta-Learning with Gaussian Processes" by Petru Tighineanu, Lukas Grossberger,
Paul Baireuther, Kathrin Skubch, Stefan Falkner, Julia Vinogradska, and Felix
Berkenkamp, which was accepted for publication at AISTATS 2024 and can be found here 
https://arxiv.org/html/2312.00742v1.
The code allows the users to reproduce and extend results reported in the study.
Please cite the above paper when reporting, reproducing or extending the results.

## Purpose of the project

This software is a research prototype, solely developed for and published as part of the
publication cited above.
It will neither be maintained nor monitored in any way.

## Setup & Run

In case you would like to install just ScaML-GP as a dependency in your Python project,
use for example
```bash
pip install git+https://github.com/boschresearch/Scalable-Meta-Learning-with-Gaussian-Processes.git
```

To run the ScaML-GP experiments, set up an environment from a clone of the repository
with `poetry install --all-extras` to include the `benchmarking` extra with the
respective dependencies.
You can then run
```bash
python scamlgp/benchmarking/configurations/branin.py submit all
python scamlgp/benchmarking/configurations/branin.py visualize all
```
to submit for example the Branin benchmark runs for ScaML-GP and visualize the results.

## Cite

In case you are using or would like to refer to ScaML-GP, please use the following
citation:
```bibtex
@article{tighineanu2024scalable,
      title={{Scalable Meta-Learning with Gaussian Processes}}, 
      author={Petru Tighineanu and Lukas Grossberger and Paul Baireuther and Kathrin Skubch and Stefan Falkner and Julia Vinogradska and Felix Berkenkamp},
      year={2024},
      journal={International Conference on Artificial Intelligence and Statistics}
}
```

## Contact

* Petru Tighineanu: petru.tighineanu@de.bosch.com
* Lukas Grossberger: lukas.grossberger@de.bosch.com

## License

`Scalable-Meta-Learning-with-Gaussian-Processes` is open-sourced under the AGPL-3.0
license.
See the [LICENSE](LICENSE) file for details.

For a list of other open source components included in
`Scalable-Meta-Learning-with-Gaussian-Processes`, see the file
[3rd-party-licenses.txt](3rd-party-licenses.txt).
