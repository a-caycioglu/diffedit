import torch
import gradio as gr
from PIL import Image
from diffedit_model import DiffEdit, load_models

# Global variables.
x = None
raw_mask = None

# Loading models.
unet, vae, tokenizer, text_encoder, scheduler = load_models()

def process_inputs(image, ref_prompt, query_prompt):
    global x, raw_mask
    if image is None:
        raise gr.Error("Please upload an image.")
    x = DiffEdit(image=image, unet=unet, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, 
                 scheduler=scheduler, query_prompt=query_prompt, ref_prompt=ref_prompt)
    raw_mask = x.mask()
    initial_mask = x.mask_binarized(raw_mask, threshold=0.35)
    return x.visualize_mask(initial_mask)

def update_mask(threshold):
    global x, raw_mask
    updated_mask = x.mask_binarized(raw_mask, threshold=threshold)
    return x.visualize_mask(updated_mask)

def generate_final_image(threshold, seed, strength):
    global x, raw_mask
    mask = x.mask_binarized(raw_mask, threshold=threshold)
    result = x.improved_masked_diffusion(prompts=[x.query_prompt, x.ref_prompt], mask=mask, 
                                         seed=int(seed), strength=strength)
    return result

def create_gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# DiffEdit Image Editor")
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(label="Original Image", type="pil")
                ref_prompt_input = gr.Textbox(label="Reference Prompt", value="Enter your reference prompt that describes original image.")
                query_prompt_input = gr.Textbox(label="Query Prompt", value="Enter your query input that describes edited image.")
                process_button = gr.Button("Generate Mask")
            
            with gr.Column(scale=1):
                mask_output = gr.Plot(label="Mask Visualization")
                threshold_slider = gr.Slider(minimum=0, maximum=1, value=0.35, step=0.01, label="Threshold")
                seed_input = gr.Number(value=10, label="Seed", precision=0)
                strength_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.5, step=0.01, label="Strength")
                generate_button = gr.Button("Generate Edited Image")
        
        output_image = gr.Image(label="Edited Image")

        process_button.click(fn=process_inputs, 
                             inputs=[image_input, ref_prompt_input, query_prompt_input],
                             outputs=mask_output)
        
        threshold_slider.change(fn=update_mask,
                                inputs=[threshold_slider],
                                outputs=mask_output)
        
        generate_button.click(fn=generate_final_image,
                              inputs=[threshold_slider, seed_input, strength_slider],
                              outputs=output_image)

    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(share=True)
