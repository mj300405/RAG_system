from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class PHI2Generator:
    def __init__(self, num_threads=None):
        # Automatically use CUDA if available, otherwise fallback to CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # If running on CPU and num_threads is specified, set the number of threads
        if self.device == "cpu" and num_threads is not None:
            torch.set_num_threads(num_threads)
        
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")
        
        # Ensure the tokenizer has a padding token, set it to EOS token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5")
        
        # Move the model to the specified device (GPU or CPU)
        self.model.to(self.device)

    def generate_answer(self, context, max_length=512):
        # Tokenize the input context and move tensors to the correct device
        inputs = self.tokenizer(context, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
      
        # Generate an answer using the model
        output_ids = self.model.generate(**inputs, max_length=max_length, num_return_sequences=1)
      
        # Decode the generated tokens to a string
        answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
      
        return answer
    
    # def generate_answer(self, context, max_length=100, continue_prompt=True):
    #     generated_text = ""
    #     while True:
    #         # Tokenize the input context (or the latest part of it) and generate answer
    #         input_ids = self.tokenizer.encode(context, return_tensors="pt").to(self.device)
    #         output_ids = self.model.generate(input_ids, max_length=max_length + len(input_ids[0]), num_return_sequences=1)
    #         part_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
    #         # Append the newly generated part to the overall generated text
    #         generated_text += part_text[len(context):]  # Append only new text
            
    #         if not continue_prompt or len(output_ids[0]) < max_length + len(input_ids[0]):
    #             break  # Stop if user does not want to continue or output is shorter than max_length
            
    #         # Ask the user if they want to continue generating text
    #         continue_gen = input("Do you want to continue generating text? (yes/no): ")
    #         if continue_gen.lower() not in ['yes', 'y']:
    #             break  # Exit the loop if the user does not want to continue
            
    #         # Use the last part of the generated text as the new context
    #         context = self.tokenizer.decode(output_ids[0][-1024:], skip_special_tokens=True)  # Example context window size

    #     return generated_text

