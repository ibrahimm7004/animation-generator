# 🌿 Ghibli-Style Animation Generator 🎨✨

This project is an AI-powered animation generator that creates frame-by-frame, Ghibli-style animations from natural language prompts. Using a combination of OpenAI GPT models, Stable Diffusion pipelines (txt2img, img2img), and ControlNet, the system produces coherent and stylized animated sequences in the spirit of Studio Ghibli’s visual aesthetic.

---

### 🎞️ Sample Output

![ghibli animation](static/animation.gif)

---

## 🚀 Features

- 🎬 **Natural Prompt to Animation**: Enter a single prompt (e.g. “a serene forest at dawn...”) and receive a full animation sequence.
- 🖼️ **Frame-by-Frame Generation**: Uses Stable Diffusion to generate the first frame from text, and `img2img` to evolve subsequent frames.
- 🧠 **LLM-Assisted Scene Breakdown**: Breaks a high-level idea into a sequence of structured animation steps using OpenAI GPT-4o or Mistral.
- 🎛️ **Style Control via LoRA + ControlNet**: Ensures stylistic consistency and smooth motion using LoRA fine-tuning and depth-based ControlNet conditioning.
- 🪄 **Smooth Transitions**: Automatically blends frames for smoother animation and less jitter using pixel-wise interpolation.
- 📽️ **GIF Output**: Compiles frames into a looping animated GIF.

---

## 📦 Tech Stack

- **Python**
- **Stable Diffusion v1.5** (`diffusers`)
- **OpenAI GPT / Mistral-7B-Instruct (via HF)**
- **ControlNet (Depth)**
- **LoRA fine-tuning**
- **Pillow**, **OpenCV**, **Torch**, **Transformers**

---

## 🧪 How It Works

### 1. Prompt Breakdown (LLM)

Uses `mistralai/Mistral-7B-Instruct-v0.3` via Hugging Face Inference API to convert a natural prompt into a Python list of frame contexts.

```python
from huggingface_hub import InferenceClient
client = InferenceClient(provider="hf-inference", api_key="hf_...")

def generate_animation_frames(prompt, steps=5):
    system_message = (
        f"You are an animation assistant to help create ghibli-themed animation frames. "
        f"Each frame must include the word 'ghibli' or describe the frame as 'ghibli-style'. "
        f"Break the given ghibli-themed idea into a smooth {steps}-step animation. "
        f"Return the result as a Python list of {steps} strings."
    )
    completion = client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content
```

### 2. Normalize Output

Mistral's output is parsed and cleaned using a custom parser to return a valid list of frames.

```python
import ast, re

def normalize_llm_frame_output(raw_output, steps=5):
    try:
        parsed = ast.literal_eval(raw_output)
        if isinstance(parsed, list):
            return parsed[:steps]
    except:
        pass
    # fallback
    pattern = re.compile(r'\d+\s*[\.\)]\s*["“]?(.*?)["”]?(?=\n\d+|\Z)', re.DOTALL)
    matches = pattern.findall(raw_output)
    return [m.strip() for m in matches[:steps]]
```

---

### 3. Generate Frames

The first frame is created using Stable Diffusion `txt2img`, and subsequent frames are generated via `img2img` using the previously generated image.

```python
def generate_frames(frames):
    for idx, frame in enumerate(frames):
        if idx == 0:
            image = pipe(frame, num_inference_steps=30).images[0]
            saved_image = save_img(image, idx)
        else:
            refined = img2img_pipe(
                prompt=frame,
                image=saved_image,
                strength=0.7,
                guidance_scale=9.0,
                num_inference_steps=40
            ).images[0]
            saved_image = save_img(refined, idx)
```

### 4. Compile to GIF

```python
from IPython.display import Image as IPyImage, display

def create_gif_from_frames(folder_path="/content/frames", output_path="/content/animation.gif", duration=300):
    frames = sorted(
        [Image.open(os.path.join(folder_path, f)) for f in os.listdir(folder_path) if f.endswith(".png")],
        key=lambda x: int(x.filename.split('_')[-1].split('.')[0])
    )
    frames[0].save(output_path, format="GIF", save_all=True, append_images=frames[1:], duration=duration, loop=0)
    display(IPyImage(filename=output_path))
```

---

## 📁 Finetuning Dataset

The Ghibli visual style was enhanced using a manually curated and captioned dataset of Ghibli-style image-caption pairs:

🔗 **Final dataset**: [ghibli-images-for-SD1.5](https://huggingface.co/datasets/ibrahim7004/ghibli-images-for-SD1.5)

📂 Dataset: 50 images with captions manually written using ChatGPT assistance for consistency

🧪 Earlier experiments:

- [lora-ghibli-images](https://huggingface.co/datasets/ibrahim7004/lora-ghibli-images)
- [lora-pak-truck-art](https://huggingface.co/datasets/ibrahim7004/lora-pak-truck-art)

Used for fine-tuning LoRA weights applied to the Stable Diffusion UNet.

---

## 🤖 Prompt Handling via LLM (Mistral)

This system uses `mistralai/Mistral-7B-Instruct-v0.3` via the Hugging Face Inference API to break a single animation idea into a coherent list of frames.

### Why Mistral?

- Strong instruction-following ability
- Fast and cost-effective via Hugging Face's hosted API
- Stable and creative for scene decomposition

### Final prompt example:

```python
system_message = (
    f"You are an animation assistant to help create ghibli-themed animation frames. "
    f"Each frame must include the word 'ghibli' or describe the frame as 'ghibli-style'. "
    f"Break the given ghibli-themed idea into a smooth 5-step animation. "
    f"Return the result as a Python list of 5 strings."
)
```

### Output Format (Example):

```python
[
  "A ghibli-style forest glows under golden sunlight.",
  "Tall ghibli trees sway gently in the wind.",
  "A ghibli cottage appears through the trees.",
  "The ghibli sky fills with birds over the valley.",
  "Sunlight fades as the ghibli village comes into view."
]
```

---

## 🎥 Example Workflow

```python
prompt = "A ghibli forest with flickering sunlight through swaying trees"
result = generate_animation_frames(prompt, steps=5)
frames = normalize_llm_frame_output(result, steps=5)
generate_frames(frames)
create_gif_from_frames()
```

---

## 📬 Contact

Created by [@ibrahim7004](https://huggingface.co/ibrahim7004) | Powered by 🤗 + 🎨 + 💻

Let me know if you want to integrate frame interpolation or music generation next!
