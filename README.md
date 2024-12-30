# Shakespearean Style Generator

This repository contains the implementation of a **Shakespearean Style Text Generator**, a natural language generation model that produces text inspired by the writing style of William Shakespeare. The project explores language modeling techniques to create poetic and dramatic sentences, suitable for creative applications.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Usage](#usage)
  - [Installation](#installation)
  - [Running the Script](#running-the-script)
- [Example Output](#example-output)
- [Dependencies](#dependencies)
- [License](#license)

## Overview
This project leverages modern natural language processing (NLP) techniques to generate text in the style of William Shakespeare. The generator is trained on Shakespeare's plays, sonnets, and other works to produce text that mimics his characteristic tone, vocabulary, and sentence structure.

## Features
- **Customizable Output:** Specify the length and starting prompt for the generated text.
- **Fine-tuned Language Model:** Uses a pre-trained transformer model fine-tuned on Shakespearean text.
- **Interactive Mode:** Allows users to input prompts interactively for on-the-fly text generation.
- **Flexible Generation Parameters:** Customize temperature, top-k, and top-p settings for diverse outputs.

## Usage

### Installation
Clone the repository to your local machine:

```bash
git clone https://github.com/sivaciov/Shakespearean-style-generator.git
cd Shakespearean-style-generator
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Running the Script
To generate text, run the `generate_text.py` script:

```bash
python generate_text.py --prompt "To be or not to be" --max_length 50 --temperature 0.8
```

#### Command-line Arguments
- `--prompt`: The starting text for the generator (default: empty string).
- `--max_length`: Maximum length of the generated text (default: 50).
- `--temperature`: Sampling temperature to control randomness (default: 1.0).
- `--top_k`: Top-k sampling for controlling vocabulary size (default: 50).
- `--top_p`: Top-p (nucleus) sampling for probability mass cutoff (default: 0.9).

### Interactive Mode
You can also run the generator in interactive mode:

```bash
python generate_text.py --interactive
```

## Example Output
Generated text example:
```
Prompt: "Shall I compare thee"

Output:
Shall I compare thee to a summer's day?
Thy beauty shines with such celestial rays.
The winds do whisper softly through the leaves,
And time doth pause, thy splendor to perceive.
```

## Dependencies
This project requires the following libraries:
- `torch`
- `transformers`
- `numpy`
- `tqdm`

Install them using:
```bash
pip install torch transformers numpy tqdm
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to contribute to the project, experiment with the model, or use it for creative purposes. Suggestions and feedback are always welcome!
