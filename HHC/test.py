import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import gradio as gr

# 模型和路径设置
model_path = 'model/LLM-Research/Meta-Llama-3___1-8B-Instruct'
lora_path = 'output/llama3_1_instruct_lora/checkpoint-699'

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
model = PeftModel.from_pretrained(model, lora_path).eval()

# 对话函数
def chat_with_model(prompt):
    messages = [
        {"role": "system", "content": "假设你是皇帝身边的女人-甄嬛。"},
        {"role": "user", "content": prompt}
    ]
    
    # 转换为输入
    input_ids = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([input_ids], return_tensors="pt").to("cuda")
    
    # 生成响应
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return response

# 使用Gradio构建界面
interface = gr.Interface(
    fn=chat_with_model,
    inputs=gr.Textbox(label="输入对话", placeholder="请输入您的问题..."),
    outputs=gr.Textbox(label="甄嬛的回答"),
    title="甄嬛对话助手"
)

# 启动应用
if __name__ == "__main__":
    interface.launch()
