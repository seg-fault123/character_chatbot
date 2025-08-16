# Naruto Dialogue Simulator

Fine-tuned LLAMA-3-8B-Instruct to mimic Naruto Uzumaki's speech style using anime transcripts and deployed an interactive chatbot interface.

## Overview

This project demonstrates the end-to-end process of building a character-driven conversational AI by:
- Processing episode transcripts of the Naruto anime from Animelon
- Extracting all Naruto-specific dialogues for dataset creation
- Fine-tuning a quantized LLAMA-3-8B-Instruct language model using LoRA adapters
- Deploying the model with a Gradio-based chatbot interface for real-time interaction

## Features

- **Authentic Dialogue:** The chatbot replicates Naruto's unique speech patterns.
- **Efficient Model Training:** Quantized to 4-bit precision using `bitsandbytes` for reduced resource usage.
- **Parameter-Efficient Tuning:** Utilizes LoRA adapters via the `peft` library.
- **Interactive Chatbot:** User-friendly Gradio interface for seamless conversations.

## Tech Stack

- **Language Model:** LLAMA-3-8B-Instruct
- **Quantization:** bitsandbytes (4-bit)
- **Fine-Tuning:** LoRA (peft library)
- **Data Processing:** Python (regex, pandas)
- **Chat Interface:** Gradio
