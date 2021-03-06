# plug-and-blend

This is the official repository for implementation of https://arxiv.org/abs/2104.04039 (Accepted at AIIDE 2021 (Oral) !!)

# What is this?

Plug-and-Blend introduce blending control to your continuation model and grants high-level planning capabilities to it. Great for generating stories, designed with co-creativity in mind and will benefit other PCG applications.

# NEW!!! Plug-and-blend as LogitsProcessor [UPDATED! ALL_IN_ONE!]
With the support of custom LogitsProcessor added to generate(), using P&B is easier than ever! Take a look at this colab notebook for a reference implementation of P&B LogitsProcessor that you can use with a single parameter update in your `generate()` !

<a href="https://colab.research.google.com/drive/1nuxJ7eGHu3WSGui3WT5cJjxR49R_Lg41?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


# If you are curious, see the demo now!
<a href="https://colab.research.google.com/github/xxbidiao/plug-and-blend/blob/main/blending_generation_demo_colab.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

(Hint: Don't forget to switch Runtime Type to GPU - Or generating a sentence will take a minute!)

# If you found this work helpful

Please consider citing this work using the following information:

```
@article{lin2021plug,
  title={Plug-and-Blend: A Framework for Controllable Story Generation with Blended Control Codes},
  author={Lin, Zhiyu and Riedl, Mark},
  journal={arXiv preprint arXiv:2104.04039},
  year={2021}
}
```
