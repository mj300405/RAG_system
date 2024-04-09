from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class PHI2Generator:
    """
    A class to generate answers using the PHI model from the Hugging Face Transformers library.
    
    This generator is designed to automatically utilize CUDA if available, falling back to CPU otherwise.
    It uses the `microsoft/phi-1_5` pretrained model for generating answers to the given context.
    
    Attributes:
        device (str): Specifies the device to use for tensor computations; 'cuda' or 'cpu'.
        tokenizer (AutoTokenizer): Tokenizer for the PHI model.
        model (AutoModelForCausalLM): The PHI generative language model.
    
    Args:
        num_threads (int, optional): The number of CPU threads to use for computation. 
                                     This is only relevant if CUDA is not available.
    """
    
    def __init__(self, num_threads=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.device == "cpu" and num_threads is not None:
            torch.set_num_threads(num_threads)
        
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5")
        self.model.to(self.device)

    def generate_answer(self, context, max_length=512):
        """
        Generates an answer for the given context using the PHI model.
        
        This method tokenizes the input context, generates an answer up to a specified length,
        and then decodes the generated tokens back to text.
        
        Args:
            context (str): The input context string to generate an answer for.
            max_length (int, optional): The maximum length of the generated answer. Defaults to 512.
        
        Returns:
            str: The generated answer text.
        """
        inputs = self.tokenizer(context, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        output_ids = self.model.generate(**inputs, max_length=max_length, num_return_sequences=1)
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

