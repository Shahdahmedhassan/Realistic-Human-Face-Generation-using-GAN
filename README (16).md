# 🧬 Realistic Human Face Generation using GAN

A Generative Adversarial Network (GAN) that learns to synthesize realistic human face images from random noise, wrapped in an interactive **Streamlit** web app.

🔗 **Live App:** [Add your deployed Streamlit link here]

---

## 📌 Overview

This project explores how GANs can generate entirely new, non-existent human faces. It trains a **Generator** and a **Discriminator** in an adversarial setup — the Generator tries to create convincing fake faces from random latent vectors, while the Discriminator tries to tell real faces from generated ones. Over training, the Generator gets better at producing realistic images.

The trained Generator is deployed in a Streamlit app where users can:
- Upload the trained Generator model
- Generate a custom number of synthetic faces on demand
- Set a random seed for reproducible results
- Explore latent-space interpolation (morphing smoothly between two generated faces)
- Download all generated faces as a `.zip` file

---

## ⚙️ What It Does

1. **Loads and preprocesses** a human face dataset (cropping, resizing, normalizing).
2. **Builds a Generator model** that maps random latent vectors (noise) into face images.
3. **Builds a Discriminator model** that classifies images as real or generated (fake).
4. **Trains both models adversarially**, monitoring generator/discriminator loss over time.
5. **Serves the trained Generator** through a Streamlit interface for on-demand face generation.

---

## 🛠️ Tools & Technologies

- **Python** — core programming language
- **TensorFlow / Keras** (or **PyTorch**) — building and training the GAN (Generator & Discriminator)
- **NumPy** — latent vector sampling and array/image processing
- **Pillow (PIL)** — image handling and export
- **Streamlit** — interactive web app / UI for the deployed model
- **Matplotlib** *(optional, for training visualization)* — plotting losses and sample outputs during training

---

## 🚀 Running the App Locally

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd <your-repo-folder>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app.py
```

Then open the app in your browser, upload your trained Generator model file, and start generating faces.

---

## 📂 Project Structure

```
├── app.py              # Streamlit application (UI + inference)
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

---

## 📈 Future Improvements

- Add support for conditional GANs (control attributes like age, expression, hairstyle)
- Improve image resolution using progressive growing or StyleGAN-based architectures
- Add an in-app training/fine-tuning mode
- Deploy with GPU-backed inference for faster generation

---

## 📄 License

This project is for educational purposes as part of an AI/Deep Learning bootcamp project.
