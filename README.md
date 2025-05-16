# 🌿 Ghibli-Style Animation Generator 🎨✨

This project is an AI-powered animation generator that creates frame-by-frame, Ghibli-style animations from natural language prompts. Using a combination of OpenAI GPT models, Stable Diffusion pipelines (txt2img, img2img), and ControlNet, the system produces coherent and stylized animated sequences in the spirit of Studio Ghibli’s visual aesthetic.

---

## 🚀 Features

- 🎬 **Natural Prompt to Animation**: Enter a single prompt (e.g. “a serene forest at dawn...”) and receive a full animation sequence.
- 🖼️ **Frame-by-Frame Generation**: Uses Stable Diffusion to generate the first frame from text, and `img2img` to evolve subsequent frames.
- 🧠 **LLM-Assisted Scene Breakdown**: Breaks a high-level idea into a sequence of structured animation steps using OpenAI GPT-4o.
- 🎛️ **Style Control via LoRA + ControlNet**: Ensures stylistic consistency and smooth motion using LoRA fine-tuning and depth-based ControlNet conditioning.
- 🪄 **Smooth Transitions**: Automatically blends frames for smoother animation and less jitter using pixel-wise interpolation.
- 📽️ **GIF Output**: Compiles frames into a looping animated GIF.

---

## 📦 Tech Stack

- **Python**
- **Stable Diffusion v1.5** (`diffusers`)
- **OpenAI GPT-4o API**
- **ControlNet (Depth, optional Segmentation)**
- **LoRA fine-tuning (Ghibli-style aesthetic)**
- **Pillow**, **OpenCV**, **Torch**, **Transformers**

---

## 🧪 How It Works

1. **Prompt Breakdown**

   - A prompt like:  
     _“A forest sways in the wind as light flickers through the trees”_
   - Is broken into a 5–10 step frame narrative using GPT-4o.

2. **Frame Generation**

   - The first frame is created with a `txt2img` pipeline and Ghibli LoRA.
   - Subsequent frames use `img2img` with the previous frame as input, guided by ControlNet (Depth).

3. **Frame Blending**

   - Intermediate blended frames are inserted between real ones for smoother transitions.

4. **GIF Assembly**
   - All frames are compiled into a `.gif` animation.

---

## 🖥️ Example

```python
prompt = "A glowing forest as the camera moves through swaying trees"
frames = generate_animation_frames_openai(prompt, steps=6)
generate_frames(frames)
create_smooth_gif()
```
