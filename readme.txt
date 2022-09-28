Very first public drop of our ui which was heavily inspired by some other work out there, there is plenty and plenty inspiration.
Since it is impossible to say who took what from whom at what time we like to say thank you to all of you.
There is one guy who insisted in having his name named even that history tells he was a thief too,
so to protect the innocent we do not name names here

All we say is something what is said to be said by many people but by one very special guy who may rest in peace and
that`s a name here it is Mr. Steve Jobs who as history tells said this:
Good artists copy and the brilliant one steal as hard as they can and get away with it.

So its up to you to think what you want.If you like the result use it, if you got any bad feelings, well dont use it then.

But we also did plenty ourself, came up with new ideas which we think others will steal or copy too very soon.
Feel free to do so but please check the license ;)

it is still under heavy development so please expect errors, hickups and headache.
If there is any error please let us know, we are happy to help and also happy to fix stuff.

If you like to contribute please get in touch we are happy to onboard you.
We are happy for any more man power to speed up development.
There is a roadmap in doing which we like to see in the real world.

If you like to add ideas please let us know too.

for now, have fun and happy renering

Let the AI dream of pixels :)

Install:

install miniconda
## Manually download 3 Model Files

**You need to get the `sd-v1-4.ckpt` file and put it on the `./models` folder first to use this. It can be downloaded from [HuggingFace](https://huggingface.co/CompVis/stable-diffusion).**

**Additionally, you should put `dpt_large-midas-2f21e586.pt` on the `./models` folder as well, [the download link is here](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt)**

**There should be another extra file `AdaBins_nyu.pt` which should be downloaded into `./pretrained` folder, [the download link is here](https://cloudflare-ipfs.com/ipfs/Qmd2mMnDLWePKmgfS8m6ntAg4nhV5VkUyAydYBp8cWWeB7/AdaBins_nyu.pt)**


run setup_env.cmd


for future use you can already download
https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth
and copy to src/gfpgan/experiments/pretrained_models

https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth
and copy to src/realesrgan/experiments/pretrained_models

Pull requests go to dev only

