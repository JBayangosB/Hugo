+++
authors = ["Jayden Bayangos-Brandt"]
title = 'Train Flux Lora'
description = "Something"
date = "2024-09-18"
draft = false
+++

# How to train a Flux LORA to make pictures of you!

I had a lot of difficulty trying to figure out how to train a LORA for flux on my system.

(4070 with 12GB VRAM)

So after a bunch of trial and error I compiled this mini tutorial with as much guidance as I needed. 

Hopefully this can help someone else out in the future!

There is probably a good chance this is out-of-date and there are better ways to do this when this is live, oh well!

## Setup

### Environment setup

Clone the following git repo with the command:

```bash
git clone -b sd3 https://github.com/kohya-ss/sd-scripts.git
```

or if that doesn't end up working due to updates. This is the commit I was checked out on when I succeeded.

```bash
git checkout 2e89cd2cc634c27add7a04c21fcb6d0e16716a2b
```

Navigate via the terminal to the newly cloned `sd-scripts` folder.

Call the commands in sequence:
```bash
python -m venv venv
```

```bash
.\venv\Scripts\activate
```

```bash
pip3 install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
```

```bash
pip install --upgrade -r requirements.txt
```

```bash
accelerate config
```

The above command will display a series of options, you should select these answers:
```
    - This machine
    - No distributed training
    - NO
    - NO
    - NO
    - all
    - fp16
```

By default inside `sd-scripts` there will be a file called `dataset.toml`, this is where you need configure the following settings:

- The resolution of the images in your dataset
- The directory of the images
- The number of repeats
- The token you want to use in prompts to trigger the LORA.

Here is what a toml might look like which you can use to update yours:

```toml
[general]
shuffle_caption = false
caption_extension = '.txt'
keep_tokens = 1

# This is a DreamBooth-style dataset
[[datasets]]
resolution = 512 # Update this with your resolution, aim for either 512 or 1024.
batch_size = 1
keep_tokens = 1

  [[datasets.subsets]]
  image_dir = 'C:\dataset\images_to_train_on' # Update this with your actual directory. 
  class_tokens = 'image name' # Update this with what you want to trigger the LORA with in your prompts.
  num_repeats = 1
```

### Dataset grooming

It is ideal that you have at least 10 photos of whatever it is your trying to train a LORA for. 

The photos should show the target with a variety of different backgrounds.

It is also a good idea to have captions for each image and they must be named in reference to the image file. Such as: `image_1.jpg` to `image_1.txt` (Tthe `.txt` file being the caption.)

You can use whatever way you desire to produce your captions, its entirely up to you and has different impacts on training.
- You can use existing tools like JoyCaption
- Just providing the token as the caption alone
  - Your training a lora for `dogs`, `dogs` is the prompt, you can caption all images with just `dogs` if you like.

## Let the training begin!

With a terminal set to the `sd-scripts` directory you can now call the following command to start it off, but you need to make sure you update parameters within the command to suit your settings.

```bash
accelerate launch --mixed_precision bf16 --num_cpu_threads_per_process 1 flux_train_network.py --pretrained_model_name_or_path C://Change//Path//To//flux1-dev.safetensors --clip_l C://Change//Path//To//clip_l.safetensors --t5xxl C://Change//Path//To//t5xxl_fp8_e4m3fn.safetensors --ae C://Change//Path//To//ae.safetensors --cache_latents_to_disk --save_model_as safetensors --sdpa --persistent_data_loader_workers --max_data_loader_n_workers 2 --seed 42 --gradient_checkpointing --mixed_precision bf16 --save_precision bf16 --network_module networks.lora_flux --network_dim 16 --network_alpha 16 --optimizer_type adafactor --learning_rate 1e-3 --network_train_unet_only --cache_text_encoder_outputs --cache_text_encoder_outputs_to_disk --fp8_base --highvram --max_train_epochs 8 --save_every_n_epochs 2 --dataset_config dataset.toml --output_dir C://Change//Path//To//Lora_Save_Directory --output_name lora_name --timestep_sampling sigmoid --model_prediction_type raw --guidance_scale 1.0 --loss_type l2 --optimizer_type adafactor --optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False" --split_mode --network_args "train_blocks=single"
```

In the above block you will need to provide paths to:
- `flux.dev.safetensors`
- `clip_l.safetensors`
- `t5xxl_fp8_e4m3fn.safetensors`
- `ae.safetensors`
- Where you want to save your LORA

The output name of your LORA.


Once you have executed the above command with your personal parameters updated, you will have kicked off the training process, this can vary greatly in time depending on the settings you selected.

Dataset count and number of repeats specificed in the `.toml` file will have the greatest impacts.

## Training done.

Congratulations! Assuming all went to plan, you will find a file named `lora-name.safetensors` inside the directory you specified above. Now you should be able to apply that LORA using the tool of your choice (E.g. ComfyUI). 

I saw decent results with a dataset of 20 diverse images with `num_repeats` set to 5. This was with caption files that only contained the name of the LORA. In the future I look forward to messing with the parameters further to find the sweet spot.

I hope this article was helpful!

### Sources

[Reddit - Flux Local LoRA Training in 16GB VRAM (quick guide in my comments)](https://www.reddit.com/r/StableDiffusion/comments/1eyr9yx/flux_local_lora_training_in_16gb_vram_quick_guide/)

[Reddit - I will train a Flux LORA for you, for free](https://www.reddit.com/r/StableDiffusion/comments/1ezd23b/i_will_train_a_flux_lora_for_you_for_free_3/)