from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

class Summarize:
    def summarize_news_descriptions(content, model_repo="anirudhsayya/t5-small-xsum-finetuned"):
        """
        Summarizes the 'description' field for all articles in news_data.
        
        Args:
            news_data: Dictionary containing 'articles' list
            model_repo: Hugging Face model repository ID
        
        Returns:
            Updated news_data with summarized descriptions
        """
        
        # Load model
        print(f"Loading model from {model_repo}...")
        model = T5ForConditionalGeneration.from_pretrained(model_repo)
        tokenizer = T5Tokenizer.from_pretrained(model_repo)
        print("Summarizer model loaded successfully on CPU!")
        
        # Prepare input
        input_text = "summarize: " + content
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        
        # Generate summary
        with torch.no_grad():
            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=64,
                min_length=10,
                num_beams=2,
                length_penalty=2.0,
                early_stopping=True
            )
        
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

