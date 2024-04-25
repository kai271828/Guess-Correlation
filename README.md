<a name="readme-top"></a>

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
<!-- [![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url] -->



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <!-- <a href="https://github.com/kai271828">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a> -->

  <h3 align="center">Guess Correlation</h3>

  <p align="center">
    Train a end-to-end model to guess corrlation from scatter plot images
    <!-- <br />
    <a href="https://github.com/kai271828"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/kai271828">View Demo</a>
    ·
    <a href="https://github.com/kai271828/.../issues">Report Bug</a>
    ·
    <a href="https://github.com/kai271828/.../issues">Request Feature</a>
  </p> -->
</div>


<!-- ### Built With -->

<!-- This section should list any major frameworks/libraries used to bootstrap your project. Leave any add-ons/plugins for the acknowledgements section. Here are a few examples. -->

<!-- * [![Python][Next.js]][Next-url]
* [![React][React.js]][React-url]
* [![Vue][Vue.js]][Vue-url]
* [![Angular][Angular.io]][Angular-url]
* [![Svelte][Svelte.dev]][Svelte-url]
* [![Laravel][Laravel.com]][Laravel-url]
* [![Bootstrap][Bootstrap.com]][Bootstrap-url]
* [![JQuery][JQuery.com]][JQuery-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- GETTING STARTED -->
## Getting Started

<!-- This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps. -->

<!-- ### Prerequisites

Install packages through pip.
* pip
  ```sh
  pip install -r requirements.txt
  ``` -->

### Installation

Install packages through pip.

1. Clone the repo
   ```sh
   git clone https://github.com/kai271828/Guess-Correlation
   ```
2. Install packages
   ```sh
   pip install -r requirements.txt
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage
Here is an example:
```sh
python guess_correlation.py \
    --annotation_file "" \
    --image_dir "" \
    --test_ratio 0.2 \
    --resize_to 150 \
    --backbone "resnet18" \
    --use_tanh True \
    --batch_size 128 \
    --optimizer "adamw" \
    --learning_rate 1e-3 \
    --weight_decay 0.1 \
    --num_epoch 50 \
    --tolerance 5 \
    --output_dir "" \
    --project_name "Guess Correlation" \
    --run_name "vit_b_16_baseline" \
    --seed 9527 \
    --do_train True \
    --do_eval True
```


<!-- ### Inference
Here is an example:
```sh
python inference.py \
    
``` -->