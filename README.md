# gpt-2
Implementation of [GPT-2](https://openai.com/blog/better-language-models/) in PyTorch.

gpt2.py contains the actual model definition, as well as a function to load pre-trained weights from Hugging Face. The attached jupyter notebooks train the model:
- train_shakespeare.ipynb trains a GPT to be character-level language model on shakespeare text, inspired by [Karpathy's minGPT](https://github.com/karpathy/minGPT/blob/master/play_char.ipynb).
- train_openwebtext10k.ipynb trains a BPE version on [openwebtext-10k](https://huggingface.co/datasets/stas/openwebtext-10k), a 10k record subset of [openwebtext](https://huggingface.co/datasets/openwebtext).

#References
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), the original GPT-2 paper from OpenAI
- Jay Alammar’s [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)
- Andrej Kaparthy’s [minGPT](https://github.com/karpathy/minGPT)
