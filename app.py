import os
import gradio as gr
import warnings
from PIL import Image
from utils import llama32_chatbot

warnings.filterwarnings('ignore')

if __name__== "__main__":
   
        # setup the gradio chatbot
        with gr.Blocks() as demo:
            gr.Markdown("## Llama_3_2 MulitModal Chatbot")
            with gr.Row():
                together_api_key = gr.Textbox(label="Together API Key", type="password", value="", placeholder="Enter your Together API key...")

            with gr.Row():
                prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here...")
            
            with gr.Row():
                images_input = gr.Image(label="Input Image", placeholder='Drag your image here...', type='filepath')

            generate_button = gr.Button("Generate")
            
            with gr.Row():
                result_text = gr.Textbox(label="Response:", lines=10, interactive=False)

            # if the model changes to "dall-e-3", we need to change the resolution and n
            generate_button.click(
                llama32_chatbot,
                inputs=[together_api_key, images_input, prompt],
                outputs=[result_text]
            )
        
            # with gr.Row():
            #     inputs_x=gr.Textbox(label="image description"),
            #     outputs_x=gr.Image(label="DALL-E Image"),
            # btn2 = gr.Button("Run")
            
            # btn2.click(fn=agentObj.generate_image, inputs=inputs_x[0], outputs=outputs_x[0])

        demo.launch(share=False)
        