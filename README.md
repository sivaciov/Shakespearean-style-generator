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
This project leverages modern natural language processing (NLP) techniques to generate text in the style of William Shakespeare. The LSTM-based generator is trained on Shakespeare's plays, sonnets, and other works to produce text that mimics his characteristic tone, vocabulary, and sentence structure.

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


## Example Output after 20 epochs
Generated text example:
```
Using device: mps
Number of unique characters: 84
Number of sequences: 5447644
X shape: (100000, 100, 84), y shape: (100000, 84)
Number of batches: 1563
Epoch 1/20, Loss: 2.2491
Epoch 2/20, Loss: 1.6079
Epoch 3/20, Loss: 1.2177
Epoch 4/20, Loss: 1.4938
Epoch 5/20, Loss: 1.0293
Epoch 6/20, Loss: 1.6013
Epoch 7/20, Loss: 1.3076
Epoch 8/20, Loss: 1.1662
Epoch 9/20, Loss: 1.1459
Epoch 10/20, Loss: 1.9947
Epoch 11/20, Loss: 0.8929
Epoch 12/20, Loss: 0.9806
Epoch 13/20, Loss: 1.8229
Epoch 14/20, Loss: 1.3903
Epoch 15/20, Loss: 0.9596
Epoch 16/20, Loss: 1.0616
Epoch 17/20, Loss: 0.8272
Epoch 18/20, Loss: 0.8964
Epoch 19/20, Loss: 0.9297
Epoch 20/20, Loss: 0.4107
To be or not to be, that is the question: :,::,::,,e:.:.,

ss
ss::::..
:.essces-sstslsefsa  sie i ls I behair in me,
  Weils men's eyes well I was tull is fliet,
  As on the old to stoph, not by diest sweet
  Crowning outling worse with a post so graces,
  But in the wrange, from my slame be not face,
  Whilst my flesh binds for bemonn every pied,
  And I your shand it her falseso to your.
  O let me love herce issancess agessed 'might,
  Which eyes invine to return of strangely?
  And ceeping of my sin, growth to sway.
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
