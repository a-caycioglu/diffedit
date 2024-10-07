# DiffEdit Project

This project implements the DiffEdit algorithm for image editing. It's based on the paper "DiffEdit: Diffusion-based semantic image editing with mask guidance" by Mostafa Dehghani, Yair Galron, Hila Chefer, Lior Wolf, and Oran Lang.

## Paper

For more details about the algorithm, please refer to the original paper:

[DiffEdit: Diffusion-based semantic image editing with mask guidance](https://arxiv.org/abs/2210.11427)

## Installation

To install the required libraries, run the following command:

```bash
pip install -r requirements.txt
```

## Running the App

This project uses Gradio to create an interactive web application. To run the app, use the following command:

```bash
python gradio_app.py
```

## Usage Instructions

1. Upload an image you want to edit.
2. Click on "Generate Mask" to create an initial mask for the image.
3. Use the threshold slider to adjust the mask. The goal is to find the optimal threshold where:
   - The mask covers the boundaries of the object you want to change.
   - The mask doesn't include too many other objects in the image.
4. Once you've found the optimal threshold, click on "Generate Edited Image" to produce the edited version of your image.
5. You can adjust the "Seed" and "Strength" values to create different variations of the edited image.

> **Note**: The quality of the edit depends on finding the right balance in the mask. Take your time to adjust the threshold for the best results. The best results would be achieved when mask is a little bigger than reference object but not contains much pixels from other objects or background of the image as mentioned in the paper.

**Feel free to experiment with different settings to achieve your desired outcome!**

---

## Without Gradio
Code is available without gradio in "diffedit.ipnb" file. 

## Limitations
The objects in reference and query prompts should be similar since it edits only mask. 

## Results
You can see both interface of gradio app and results of the model.

![](https://github.com/a-caycioglu/diffedit/blob/main/results-on-gradio.png?raw=true&width=100)

## Furthermore
You can experiment with different pretrained models or schedulers. It can effect quality of results. Also, the binarized mask can be used as the input mask of a Inpainting pipeline(StableDiffusionInpaintPipeline) from diffusers library.

### Additional Information

- The app uses a Gradio interface for easy interaction.
- Make sure to adjust the mask carefully for the best editing results.
- Experiment with different seed and strength values for various outcomes.
- It is designed to work with 512x512 input image size.

