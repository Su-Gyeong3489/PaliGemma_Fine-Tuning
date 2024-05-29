# Install libraries.  
###### Install the latest ver. of transformer library

    pip install git+https://github.com/huggingface/transformers.git
    #pip install -q -U git+https://github.com/huggingface/transformers.git@paligemma_fix_bf16_multigpu
    
    pip install pillow
    pip install sentencepiece
    pip install --upgrade torch torchvision
    pip install tensorboardX
    
    pip install -q -U accelerate bitsandbytes git+https://github.com/huggingface/transformers.git
    pip install datasets -q
    pip install peft -q


# Fine-tuning

    python3 paligemma_fine-tuning.py

**** ** **
###### Optimizer Options
Choose one of them for optimizer and modify Line 90 of paligemma_fine-tuning.py.  
(In case of QLoRA fine-tuning, choose "paged_adamw_8bit" for optimizer.)  

"adamw_hf", "paged_adamw_8bit", "adamw_torch", "adafactor", "adamw_torch_fused", "adafactor_hf"  
  
e.g.) optim="paged_adamw_8bit",  
**** ** **

    
# Reference
  <https://huggingface.co/blog/paligemma>
